from __future__ import annotations

import random
import re
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import psycopg2
import requests
from airflow.models import Variable
from airflow.sdk import DAG, task
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

CONFIG = Variable.get("crawl_job", deserialize_json=True)

# BASE = "https://www.topcv.vn"
HEADERS = {
    # giữ UA thật; có thể xoay vòng nếu cần
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.topcv.vn/",
    "Connection": "keep-alive",
}

# DB_NAME = "job_database"
# DB_USER = "postgres"
# DB_PASSWORD = "postgres"
# DB_HOST = "localhost"
# DB_PORT = "5432"

# TABLE_NAME = "job_demo"

# NUM_PAGES = 1

BASE = CONFIG["BASE"]

DB_NAME = CONFIG["DB_NAME"]
DB_USER = CONFIG["DB_USER"]
DB_PASSWORD = CONFIG["DB_PASSWORD"]
DB_HOST = CONFIG["DB_HOST"]
DB_PORT = CONFIG["DB_PORT"]
TABLE_NAME = CONFIG["TABLE_NAME"]
NUM_PAGES = int(CONFIG["NUM_PAGES"])
SCHEDULE = CONFIG["SCHEDULE"]
MAX_ACTIVATE_RUN = int(CONFIG["max_active_runs"])
CUR_RATE = float(CONFIG["cur_rate"])

VALID_TITLES = CONFIG["valid_titles"]
VALID_SKILLS = CONFIG["valid_skills"]
VALID_INDUSTRIES = CONFIG["valid_industries"]
MAPPING_LOCATION = CONFIG["mapping_location"]

VALID_COLS = [
    "title",
    "role",
    "job_url",
    "company",
    "company_size",
    "company_url",
    "company_industry",
    "company_address",
    "company_description",
    "salary_list",
    "salary_min",
    "salary_max",
    "salary_avg",
    "location_city",
    "experience_required",
    "deadline",
    "tags",
    "working_addresses",
    "working_times",
    "job_description",
    "job_requirement",
    "benefits",
    "crawled_at",
    "main_location",
    "industry",
    "all_skills",
]


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)

    # Retry cho lỗi tạm thời và 429
    retry = Retry(
        total=6,
        connect=3,
        read=3,
        status=6,
        backoff_factor=1.2,  # backoff cơ bản
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,  # tôn trọng Retry-After
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # pre-warm cookie (nhiều site set cookie/anti-bot ở trang chủ)
    try:
        s.get(BASE, timeout=20)
        time.sleep(1.0)
    except requests.RequestException:
        pass
    return s


def text(el) -> Optional[str]:
    if not el:
        return None
    t = el.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", t) if t else None


def smart_sleep(min_s=1.2, max_s=2.8):
    # nghỉ ngẫu nhiên để “giống người”
    time.sleep(random.uniform(min_s, max_s))


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    # vòng lặp thủ công để xử lý 429 với jitter bổ sung
    for attempt in range(1, 6):
        r = session.get(url, timeout=30)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = int(retry_after)
                except ValueError:
                    wait = 6 * attempt
            else:
                wait = 6 * attempt
            # jitter
            wait = wait + random.uniform(0.5, 2.0)
            print(f"[WARN] 429 tại {url} → ngủ {wait:.1f}s (attempt {attempt})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    # lần cuối: raise
    r.raise_for_status()
    return BeautifulSoup("", "lxml")


# ------------ Search page ------------
def parse_search_page(session: requests.Session, url: str) -> List[Dict]:
    try:
        soup = get_soup(session, url)
        jobs = []
        for job in soup.select("div.job-item-search-result"):
            a_title = job.select_one("h3.title a[href]")
            if not a_title:
                continue
            title = text(a_title)
            job_url = urljoin(BASE, a_title.get("href"))

            comp_a = job.select_one("a.company[href]")
            company = text(job.select_one("a.company .company-name"))
            company_url = urljoin(BASE, comp_a.get("href")) if comp_a else None

            salary = text(job.select_one("label.title-salary"))
            address = text(job.select_one("label.address .city-text"))
            exp = text(job.select_one("label.exp span"))

            jobs.append(
                {
                    "title": title,
                    "job_url": job_url,
                    "company": company,
                    "company_url": company_url,
                    "salary_list": salary,
                    "location_city": address,
                    "experience_required": exp,
                }
            )
        return jobs
    except:
        print(traceback.format_exc())
        return []


# ------------ Job detail page ------------
def pick_info_value(soup: BeautifulSoup, title: str) -> Optional[str]:
    for sec in soup.select(".job-detail__info--section"):
        t = text(sec.select_one(".job-detail__info--section-content-title")) or ""
        if t.lower() == title.lower():
            v = sec.select_one(".job-detail__info--section-content-value")
            return text(v) if v else text(sec)
    return None


def extract_deadline(soup: BeautifulSoup) -> Optional[str]:
    for el in soup.select(
        ".job-detail__info--deadline, .job-detail__information-detail--actions-label"
    ):
        t = text(el)
        if t and "Hạn nộp" in t:
            m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", t)
            return m.group(1) if m else t
    return None


def extract_tags(soup: BeautifulSoup):
    return [text(a) for a in soup.select(".job-tags a.item") if text(a)]


def extract_desc_blocks(soup: BeautifulSoup):
    data = {}
    for item in soup.select(".job-description .job-description__item"):
        h3 = text(item.select_one("h3")) or ""
        content = item.select_one(".job-description__item--content")
        if content:
            data[h3] = text(content)
    return data


def extract_working_addresses(soup: BeautifulSoup):
    out = []
    for item in soup.select(".job-description__item h3"):
        if "Địa điểm làm việc" in (text(item) or ""):
            wrap = item.find_parent(class_="job-description__item")
            if wrap:
                for d in wrap.select(
                    ".job-description__item--content div, .job-description__item--content li"
                ):
                    val = text(d)
                    if val:
                        out.append(val)
    return out


def extract_working_times(soup: BeautifulSoup):
    out = []
    for item in soup.select(".job-description__item h3"):
        if "Thời gian làm việc" in (text(item) or ""):
            wrap = item.find_parent(class_="job-description__item")
            if wrap:
                for d in wrap.select(
                    ".job-description__item--content div, .job-description__item--content li"
                ):
                    val = text(d)
                    if val:
                        out.append(val)
    return out


def extract_company_link_from_job(soup: BeautifulSoup) -> Optional[str]:
    cand = soup.select_one("a.company[href]") or soup.select_one("a[href*='/cong-ty/']")
    return urljoin(BASE, cand["href"]) if cand and cand.has_attr("href") else None


def scrape_job_detail(session: requests.Session, job_url: str) -> Dict:
    soup = get_soup(session, job_url)
    smart_sleep()  # nghỉ nhẹ giữa các trang

    title = text(soup.select_one(".job-detail__info--title, h1"))
    salary = pick_info_value(soup, "Mức lương")
    location = pick_info_value(soup, "Địa điểm")
    experience = pick_info_value(soup, "Kinh nghiệm")
    deadline = extract_deadline(soup)
    tags = extract_tags(soup)
    desc_blocks = extract_desc_blocks(soup)
    addrs = extract_working_addresses(soup)
    times = extract_working_times(soup)
    company_url_detail = extract_company_link_from_job(soup)

    return {
        "detail_title": title,
        "detail_salary": salary,
        "detail_location": location,
        "detail_experience": experience,
        "deadline": deadline,
        "tags": "; ".join(tags) if tags else None,
        "job_description": desc_blocks.get("Mô tả công việc"),
        "job_requirement": desc_blocks.get("Yêu cầu ứng viên"),
        "benefits": desc_blocks.get("Quyền lợi"),
        "working_addresses": "; ".join(addrs) if addrs else None,
        "working_times": "; ".join(times) if times else None,
        "company_url_from_job": company_url_detail,
    }


# ------------ Company page ------------
def scrape_company(session: requests.Session, company_url: Optional[str]) -> Dict:
    if not company_url:
        return {
            "company_name_full": None,
            "company_website": None,
            "company_size": None,
            "company_industry": None,
            "company_address": None,
            "company_description": None,
        }
    soup = get_soup(session, company_url)
    smart_sleep()

    # name
    company_name = None
    for css in [
        "h1.company-name",
        "h1.title",
        "div.company-header h1",
        "div.company-info h1",
        "meta[property='og:title']",
        "meta[property='og:site_name']",
        "title",
    ]:
        el = soup.select_one(css)
        if el:
            company_name = el.get("content") if el.name == "meta" else text(el)
            if company_name:
                company_name = re.sub(r"\s*\|\s*TopCV.*$", "", company_name, flags=re.I)
                break

    website = size = industry = address = None
    containers = [
        "div.company-overview",
        "div.company-detail",
        "div.company-profile",
        "section#company",
        "section.company-info",
        "div.box-intro-company",
        "div.company-info-container",
    ]
    container = None
    for css in containers:
        c = soup.select_one(css)
        if c:
            container = c
            break
    if container is None:
        container = soup

    rows = container.select(
        "li, .row, .item, .info-item, .company-info-item, .dl, .d-flex"
    )
    for row in rows:
        row_text = text(row) or ""
        label = None
        value = None
        strong = row.find(["strong", "b"])
        if strong:
            label = text(strong)
            value = row_text
            if label:
                value = re.sub(re.escape(label), "", value, flags=re.I).strip(" :-–—")
        else:
            m = re.match(r"^([^:：]+)[:：]\s*(.+)$", row_text)
            if m:
                label, value = m.group(1).strip(), m.group(2).strip()

        if not label or not value:
            continue

        ln = re.sub(r"\s+", " ", label.lower())
        if "website" in ln or "trang web" in ln:
            website = value
        elif "quy mô" in ln or "size" in ln or "nhân sự" in ln:
            size = value
        elif "lĩnh vực" in ln or "industry" in ln or "ngành" in ln:
            industry = value
        elif "địa chỉ" in ln or "address" in ln:
            address = value

    description = None
    for css in [
        "div.company-description",
        "div#company-description",
        "div.box-intro-company",
        "div.company-introduction",
        "div.description",
        "section.company-description",
        "div#readmore-company",
        "div#readmore-content",
    ]:
        el = soup.select_one(css)
        if el:
            description = text(el)
            if description:
                break

    return {
        "company_name_full": company_name,
        "company_website": website,
        "company_size": size,
        "company_industry": industry,
        "company_address": address,
        "company_description": description,
    }


@task
def crawl_data(
    start_page: int = 1, end_page: int = NUM_PAGES, delay_between_pages=(0.5, 1)
):
    rows: List[Dict] = []
    seen_jobs = set()

    s = build_session()

    for page in tqdm(range(start_page, end_page + 1)):
        cur_count = len(rows)
        url = f"https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257?type_keyword=1&page={page}&category_family=r257&sba=1"
        print(f"[INFO] Crawling search page {page}: {url}")
        jobs = parse_search_page(s, url)

        if not jobs:
            print(f"[INFO] Trang {page} không còn job — dừng sớm.")
            break

        for j in jobs:
            job_url = j["job_url"]
            job_id = urlparse(job_url).path
            if job_id in seen_jobs:
                continue
            seen_jobs.add(job_id)

            # chi tiết job
            try:
                detail = scrape_job_detail(s, job_url)
            except Exception as e:
                print(f"[WARN] Lỗi job detail {job_url}: {e}")
                detail = {
                    k: None
                    for k in [
                        "detail_title",
                        "detail_salary",
                        "detail_location",
                        "detail_experience",
                        "deadline",
                        "tags",
                        "job_description",
                        "job_requirement",
                        "benefits",
                        "working_addresses",
                        "working_times",
                        "company_url_from_job",
                    ]
                }

            company_url = detail.get("company_url_from_job") or j.get("company_url")

            # chi tiết công ty
            try:
                comp = scrape_company(s, company_url)
            except Exception as e:
                print(f"[WARN] Lỗi company {company_url}: {e}")
                comp = {
                    k: None
                    for k in [
                        "company_name_full",
                        "company_website",
                        "company_size",
                        "company_industry",
                        "company_address",
                        "company_description",
                    ]
                }

            row = {**j, **detail, **comp}
            rows.append(row)
        print(f"count jobs: {len(rows)}")
        # nghỉ giữa các trang (random)
        if len(rows) == cur_count:
            break
        smart_sleep(*delay_between_pages)
    return rows


INVERT_INDUSTRIES = {vv: k for k, v in VALID_INDUSTRIES.items() for vv in v}


def match_keyword(text, kw):
    kw_lower = kw.lower().strip()
    pattern = r"\b" + re.escape(kw_lower) + r"\b"

    # Case: For very short tokens (1–2 letters), require stricter spacing or punctuation context
    if len(kw_lower) <= 2:
        pattern = rf"(?<!\w){re.escape(kw_lower)}(?!\w)"

    if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
        return True

    return False


def extract_skills(text, skills=VALID_SKILLS):
    if isinstance(text, str):
        text = text.lower()
        found_skills = set()

        for kw in skills:
            is_match = match_keyword(text, kw)
            if is_match:
                found_skills.add(kw)

        return sorted(found_skills)

    return []


def extract_title(text, title_dict=VALID_TITLES):
    match_values = {}
    if isinstance(text, str):
        text = text.lower()
        for title, keywords in title_dict.items():
            for kw in keywords:
                is_match = match_keyword(text, kw)
                if is_match:
                    match_values[title] = kw
                    break

        if match_values:
            return max(match_values, key=lambda k: len(match_values[k]))

        return "Other"
    return None


def extract_industry(text):
    if isinstance(text, str):
        return INVERT_INDUSTRIES.get(text, "Other")
    return None


def parse_salary(s, cur_rate=CUR_RATE):
    """
    Parse salary strings like:
    '20 - 30 triệu', 'Up to 1000 USD', '800 - 3,500 USD'
    → returns (salary_min, salary_max, currency) in actual numeric values
    """
    if not isinstance(s, str):
        return np.nan, np.nan, None

    s = s.lower().strip()
    salary_min = salary_max = np.nan
    currency = None

    # Detect currency
    if "triệu" in s:
        multiplier = 1_000_000
    elif "usd" in s or "$" in s:
        multiplier = cur_rate
    else:
        multiplier = 1  # fallback

    # Handle negotiable cases
    if any(word in s for word in ["thoả", "thỏa", "negotiable", "thoa thuan"]):
        return np.nan, np.nan, currency

    # Normalize numbers: remove commas used as thousand separators
    s_clean = re.sub(r"(?<=\d),(?=\d)", "", s)  # 3,500 → 3500

    # Extract numeric values (with decimals)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s_clean)]
    for i in nums:
        if i > 100000:
            multiplier = 1000000
            break

    # Determine salary range
    if len(nums) == 2:
        salary_min, salary_max = nums
    elif len(nums) == 1:
        if any(word in s for word in ["tới", "đến", "up to", "upto", "đến mức"]):
            salary_min, salary_max = np.nan, nums[0]
        elif any(word in s for word in ["từ", "from"]):
            salary_min, salary_max = nums[0], np.nan
        else:
            salary_min = salary_max = nums[0]

    # Convert to actual value (VND or USD)
    if not np.isnan(salary_min):
        salary_min *= multiplier
    if not np.isnan(salary_max):
        salary_max *= multiplier

    if not (np.isnan(salary_min)) and not np.isnan(salary_max):
        salary_mean = (salary_min + salary_max) / 2
    elif not np.isnan(salary_min):
        salary_mean = salary_min
    elif not np.isnan(salary_max):
        salary_mean = salary_max
    else:
        salary_mean = None

    return salary_min, salary_max, salary_mean


def extract_main_location(text):
    for k, v in MAPPING_LOCATION.items():
        for loc in v:
            if loc.lower() in text.lower():
                return k
    return text.split(",")[0].split(";")[0].split("và")[0].split("&")[0].strip()


def normalize_job_url(x):
    return x.split(".html")[0] + ".html"


@task
def transform_and_insert_data(raw_data):
    df = pd.DataFrame(raw_data)
    df["job_url"] = df["job_url"].apply(lambda x: normalize_job_url(x))
    df[["salary_min", "salary_max", "salary_avg"]] = df["salary_list"].apply(
        lambda x: pd.Series(parse_salary(x))
    )
    df["role"] = df["title"].apply(lambda x: extract_title(x))
    df["industry"] = df["role"].apply(lambda x: extract_industry(x))
    df["job_requirement"] = df["job_requirement"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["all_skills"] = df[["job_requirement", "title", "tags"]].apply(
        lambda x: extract_skills(" ".join(x)), axis=1
    )
    df["main_location"] = df["location_city"].apply(lambda x: extract_main_location(x))
    df["crawled_at"] = datetime.now()
    df["deadline"] = pd.to_datetime(df["deadline"], dayfirst=True)
    df["deadline"] = df["deadline"].fillna(datetime.now() + timedelta(days=30))
    norm_data = []
    for dt in df.to_dict(orient="records"):
        temp = []
        for c in VALID_COLS:
            if c in ("deadline", "crawled_at"):
                temp.append(str(dt.get(c)))
            else:
                temp.append(dt.get(c, None))
        norm_data.append(temp)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    cursor = conn.cursor()

    insert_query = f"""
    INSERT INTO  {TABLE_NAME}(
        title, role, job_url, company, company_size, company_url,
       company_industry, company_address, company_description,
       salary_list, salary_min, salary_max, salary_avg,
       location_city, experience_required, deadline, tags,
       working_addresses, working_times, job_description,
       job_requirement, benefits, crawled_at, main_location,
       industry, all_skills
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (job_url) DO NOTHING;
    """

    cursor.executemany(insert_query, norm_data)
    conn.commit()

    print(f"✅ {len(norm_data)} rows inserted (or updated) successfully!")

    cursor.close()
    conn.close()


with DAG(
    dag_id="crawl_job",
    description="This DAG runs a workflow for job crawling from TOPCV",
    schedule=SCHEDULE,
    start_date=datetime(2021, 1, 1),
    max_active_runs=MAX_ACTIVATE_RUN,
    catchup=False,
    tags=["bigdata"],
) as dag:

    transform_and_insert_data(crawl_data())
