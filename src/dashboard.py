#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import traceback

import docx
import gradio as gr
import numpy as np
import pandas as pd
import PyPDF2
from google import genai
from google.genai import types

os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
MODEL_NAME = "gemini-2.5-flash"


# === CONFIGURATION ===
DB_NAME = "job_database"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

TABLE_NAME = "job"  # name for new table
DEFAULT_QUERY = f"SELECT * FROM {TABLE_NAME} WHERE status='open';"


# Auto-install dependencies
def install_dependencies():
    """Automatically install required packages"""
    packages = [
        "plotly",
        "pandas",
        "openpyxl",
        "wordcloud",
        "matplotlib"
    ]

    for package in packages:
        try:
            __import__(package if package != "openpyxl" else "openpyxl")
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"]
            )


install_dependencies()

import datetime
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

import gradio as gr
from sqlalchemy import create_engine

HEIGHT = 400
COL_WIDTHS = [275, 275, 75, 75, 100, 100]
NUM_SCALE = 7

PRIMARY_GRADIENT = "#667eea"
SECONDARY_GRADIENT = "#764ba2"
ACCENT_COLOR = "#FF6B6B"

custom_css = f"""
    <style>
    :root {{
        --primary-color: {PRIMARY_GRADIENT};
        --secondary-color: {SECONDARY_GRADIENT};
        --accent-color: {ACCENT_COLOR};
    }}

    .main {{
        padding: 0rem 0rem;
    }}

    .gradient-header {{
        background: linear-gradient(135deg, {PRIMARY_GRADIENT} 0%, {SECONDARY_GRADIENT} 100%);
        padding: 0px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}

    table td {{
    max-width: 400px;
    overflow: auto;
    white-space: nowrap;
    font-size: 10pt;
    }}


    .gradio-container input[type="number"],
    .gradio-container input[type="text"],
    .gradio-container textarea,
    .gradio-container .input-text {{
        font-size: 24px !important; /* Adjust this value to your desired size */
        height: 80px !important;
        font-weight: bold !important;
        line-height: 80px !important; 
        border: none !important;
        box-shadow: none !important; /* Remove any shadow that might look like a border */
    }}
    

    .stMetric {{
        background-color: rgba(102, 126, 234, 0.05);
        padding: 0px;
        border-radius: 8px;
        border-left: 4px solid {PRIMARY_GRADIENT};
    }}

    .section-title {{
        background: linear-gradient(90deg, {PRIMARY_GRADIENT} 0%, {SECONDARY_GRADIENT} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .gradio-container {{
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 0px; /* optional spacing */
    }}

    .filter-box {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 0px;
        border-radius: 10px;
        border-left: 4px solid {PRIMARY_GRADIENT};
        margin-bottom: 10px;
    }}
    </style>
"""


# ============================================================================
# DATA LOADING & CACHING
# ============================================================================
def load_data(query=DEFAULT_QUERY):
    """Load and process job data from PostgreSQL"""

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    df = pd.read_sql(query, engine)
    engine.dispose()

    return df


def create_query_time(st, end):
    return f"SELECT * FROM {TABLE_NAME} WHERE crawled_at >= '{st}' AND crawled_at < '{end}';"


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_salary_by_industry(filtered_df):
    """Average salary by industry - FIXED HEIGHT"""
    if filtered_df.empty or filtered_df["salary_avg"].notna().sum() == 0:
        return None

    salary_data = (
        filtered_df.groupby("industry")["salary_avg"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig = px.bar(
        x=salary_data.values,
        y=salary_data.index,
        orientation="h",
        title="💰 Average Salary by Industry",
        labels={"x": "Average Salary (triệu VND)", "y": "Industry"},
        color=salary_data.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=HEIGHT, showlegend=False, margin=dict(l=120, r=50, t=50, b=50)
    )
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def create_jobs_by_location(filtered_df):
    """Jobs by location"""
    if filtered_df.empty:
        return None

    location_data = filtered_df["main_location"].value_counts().head(10)

    fig = px.bar(
        x=location_data.index,
        y=location_data.values,
        title="📍 Job Postings by Location",
        labels={"x": "Location", "y": "Number of Jobs"},
        # color=location_data.values,
        # color_continuous_scale="Viridis",
        color_discrete_sequence=["yellow"],
    )
    fig.update_layout(height=HEIGHT, showlegend=False, xaxis_tickangle=-45)
    return fig


def create_industry_distribution(filtered_df):
    """Industry distribution"""
    if filtered_df.empty:
        return None

    industry_data = filtered_df["industry"].value_counts()
    palette = px.colors.qualitative.Light24
    palette = (palette * 5)[: len(industry_data)]

    fig = px.pie(
        values=industry_data.values,
        names=industry_data.index,
        title="🏭 Job Distribution by Industry",
        hole=0.4,
        color_discrete_sequence=palette,
    )
    fig.update_layout(height=HEIGHT)
    return fig


def create_experience_chart(filtered_df):
    """Experience level distribution"""
    if filtered_df.empty:
        return None

    exp_data = filtered_df["experience_required"].value_counts()

    fig = px.bar(
        x=exp_data.index,
        y=exp_data.values,
        title="💼 Jobs by Experience Level",
        labels={"x": "Experience", "y": "Number of Jobs"},
        color_discrete_sequence=["darkgreen"],  # solid green bars
    )
    fig.update_layout(height=HEIGHT, showlegend=False)
    return fig


def create_salary_distribution(filtered_df):
    """Salary range distribution - COMPACT"""
    if filtered_df.empty or filtered_df["salary_avg"].notna().sum() == 0:
        return None

    salary_data = filtered_df[filtered_df["salary_avg"].notna()]["salary_avg"]
    salary_filtered = salary_data[salary_data <= salary_data.quantile(0.95)]

    fig = px.histogram(
        salary_filtered,
        nbins=25,
        title="💵 Salary Distribution",
        labels={"value": "Salary (triệu VND)", "count": "Number of Jobs"},
        color_discrete_sequence=["#FF6B6B"],
    )
    fig.update_layout(height=HEIGHT, showlegend=False)
    return fig


def create_top_companies(filtered_df):
    """Top hiring companies"""
    if filtered_df.empty:
        return None

    companies = filtered_df["company"].value_counts().head(12)

    fig = px.bar(
        x=companies.values,
        y=companies.index,
        orientation="h",
        title="🏢 Top Companies by Job Postings",
        labels={"x": "Number of Jobs", "y": "Company"},
        color=companies.values,
        color_continuous_scale="Oranges",
    )
    fig.update_layout(
        height=HEIGHT, showlegend=False, margin=dict(l=275, r=50, t=50, b=50)
    )
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def create_wordcloud_compact(filtered_df, height=400, width=1000):
    """Create compact word cloud from skills"""
    if filtered_df.empty:
        return None

    all_skills = []
    for skills in filtered_df["all_skills"]:
        all_skills.extend(skills)

    if not all_skills:
        return None

    skill_freq = Counter(all_skills)

    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color="white",
            colormap="viridis",
            relative_scaling=0.5,
            min_font_size=8,
        ).generate_from_frequencies(skill_freq)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        return fig
    except Exception as e:
        return None


# ============================================================================
# PAGE 1: MARKET OVERVIEW
# ============================================================================


def cal_mean_format(df, col):
    val = df[col].mean() if df[col].notna().sum() > 0 else 0
    return f"{val/1e6:.1f}M" if val > 0 else "N/A"


def create_job_list_table(df):
    cols = [
        "title",
        "company",
        "main_location",
        "salary_avg",
        "experience_required",
        "industry",
    ]

    show_df = df[cols].copy()
    show_df.columns = [
        "Job Title",
        "Company",
        "Location",
        "Salary (M)",
        "Experience",
        "Industry",
    ]
    show_df["Salary (M)"] = show_df["Salary (M)"].apply(
        lambda x: f"{x/1e6:.1f}" if pd.notna(x) else "N/A"
    )
    return show_df


def apply_filters(industry, location):
    df = load_data()
    filtered = df.copy()

    if industry != "All":
        filtered = filtered[filtered["industry"] == industry]

    if location != "All":
        filtered = filtered[filtered["main_location"] == location]

    # Metrics
    total_jobs = len(filtered)
    companies = filtered["company"].nunique()

    avg_min_sal = cal_mean_format(filtered, "salary_min")
    avg_max_sal = cal_mean_format(filtered, "salary_max")

    # Charts
    fig_location = create_jobs_by_location(filtered)
    fig_industry = create_industry_distribution(filtered)
    fig_exp = create_experience_chart(filtered)
    fig_salary = create_salary_distribution(filtered)
    fig_sal_industry = create_salary_by_industry(filtered)
    fig_top_company = create_top_companies(filtered)
    fig_wordcloud = create_wordcloud_compact(filtered, height=400, width=1000)

    # Table
    show_df = create_job_list_table(filtered)

    return (
        total_jobs,
        companies,
        avg_min_sal,
        avg_max_sal,
        fig_location,
        fig_industry,
        fig_exp,
        fig_salary,
        fig_sal_industry,
        fig_top_company,
        fig_wordcloud,
        show_df,
    )


def apply_filters_analysis(industry, locations, time_start, time_end):
    query = create_query_time(time_start, time_end)
    df = load_data(query)

    filtered = df[df["industry"] == industry].copy()

    if locations:
        filtered = filtered[filtered["main_location"].isin(locations)]

    if filtered.empty:
        return (
            "No results",
            0,
            "N/A",
            "N/A",
            "N/A",
            None,
            None,
            None,
            None,
            None,
            None,
            pd.DataFrame(),
        )

    # ---- METRICS ----
    jobs = len(filtered)
    companies = filtered["company"].nunique()
    avg_sal_fmt = cal_mean_format(filtered, "salary_avg")

    top_loc = (
        filtered["main_location"].value_counts().index[0]
        if not filtered.empty
        else "N/A"
    )
    top_exp = (
        filtered["experience_required"].value_counts().index[0]
        if not filtered.empty
        else "N/A"
    )

    # ---- Skills Extraction ----
    all_skills = []
    for skills in filtered["all_skills"]:
        all_skills.extend(skills)

    skill_freq = Counter(all_skills)
    top_skills = dict(skill_freq.most_common(15))

    # ---- Skill Bar Chart ----
    if top_skills:
        fig_skills = px.bar(
            x=list(top_skills.values()),
            y=list(top_skills.keys()),
            orientation="h",
            title=f"Top 15 Skills – {industry}",
            labels={"x": "Frequency", "y": "Skill"},
            color=list(top_skills.values()),
            color_continuous_scale="Blues",
        )
        fig_skills.update_layout(height=HEIGHT, showlegend=False)
    else:
        fig_skills = None

    # ---- Skill Table ----
    skills_df = pd.DataFrame(skill_freq.most_common(20), columns=["Skill", "Frequency"])

    # ---- Wordcloud ----
    fig_wc = create_wordcloud_compact(filtered, height=400, width=1000)

    # ---- Job Title Chart ----
    titles = filtered["title"].value_counts().head(15)
    fig_titles = px.bar(
        x=titles.values,
        y=titles.index,
        orientation="h",
        title="Top Job Titles",
        labels={"x": "Count", "y": "Job Title"},
        color=titles.values,
        color_continuous_scale="Greens",
    )
    fig_titles.update_layout(height=HEIGHT, showlegend=False, margin=dict(l=200))
    fig_titles.update_yaxes(tickfont=dict(size=10))

    # ---- Location Pie ----
    loc = filtered["main_location"].value_counts()
    palette = px.colors.qualitative.Light24
    palette = (palette * 5)[: len(loc)]

    fig_location = px.pie(
        values=loc.values,
        names=loc.index,
        title="Location Distribution",
        hole=0.4,
        color_discrete_sequence=palette,
    )
    fig_location.update_layout(height=HEIGHT)

    # ---- Salary By Title ----
    salary_by_title = (
        filtered.groupby("title")["salary_avg"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    if len(salary_by_title) > 0:
        fig_salary_title = px.bar(
            x=salary_by_title.values,
            y=salary_by_title.index,
            orientation="h",
            title="Salary by Job Title",
            labels={"x": "Avg Salary (M)", "y": "Job Title"},
            color=salary_by_title.values,
            color_continuous_scale="RdYlGn",
        )
        fig_salary_title.update_layout(
            height=HEIGHT, showlegend=False, margin=dict(l=250)
        )
        fig_salary_title.update_yaxes(tickfont=dict(size=10))
    else:
        fig_salary_title = None

    # ---- Salary Distribution ----
    fig_salary_dist = create_salary_distribution(filtered)

    # ---- Table ----
    table_cols = [
        "title",
        "company",
        "main_location",
        "salary_avg",
        "experience_required",
    ]
    table_df = filtered[table_cols].copy()
    table_df.columns = [
        "Job Title",
        "Company",
        "Location",
        "Salary (M)",
        "Experience",
    ]
    table_df["Salary (M)"] = table_df["Salary (M)"].apply(
        lambda x: f"{x/1e6:.1f}" if pd.notna(x) else "N/A"
    )

    return (
        f"{industry}",
        jobs,
        companies,
        avg_sal_fmt,
        top_loc,
        top_exp,
        fig_skills,
        skills_df,
        fig_wc,
        fig_titles,
        fig_location,
        fig_salary_title,
        fig_salary_dist,
        table_df,
    )


# Functions
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text(file):
    """Extract text based on file type"""
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""


def llm_match_job(cv, jd):
    try:
        # Initialize the client
        client = genai.Client()
        json_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "overall_score": types.Schema(
                    type=types.Type.NUMBER,
                    description="Compatibility score between Job Description (JD) and CV, on a scale of 100.",
                ),
                "matched_skills": types.Schema(
                    type=types.Type.ARRAY,
                    description="List of skills from the CV that match the JD.",
                    items=types.Schema(type=types.Type.STRING),
                ),
                "missing_skills": types.Schema(
                    type=types.Type.ARRAY,
                    description="List of required skills from the JD that are missing in the CV.",
                    items=types.Schema(type=types.Type.STRING),
                ),
                "cv_years": types.Schema(
                    type=types.Type.NUMBER,
                    description="Total years of professional experience found in the CV.",
                ),
                "cv_job_title": types.Schema(
                    type=types.Type.STRING,
                    description="The most recent or relevant job title found in the CV.",
                ),
                "jd_years": types.Schema(
                    type=types.Type.NUMBER,
                    description="Minimum years of experience required by the Job Description.",
                ),
                "review": types.Schema(
                    type=types.Type.STRING,
                    description="General review, including points for CV improvement, written entirely in Vietnamese (bằng tiếng Việt).",
                ),
            },
            # Ensure all fields are included in the output
            required=[
                "overall_score",
                "matched_skills",
                "missing_skills",
                "cv_years",
                "cv_job_title",
                "jd_years",
                "review",
            ],
        )

        # 2. Configure the API call for JSON output
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
        )

        # 1. Define your prompt
        prompt = """Bạn là một nhà tuyển dụng, bạn có một job description và nhận được một CV. Hãy phân tích độ thích hợp của CV với vị trí đang tuyển dụng.
        Trả về dưới dạng json: 
            {"overall_score": <tương thích giữa JD và CV, theo thang điểm 100>,
            "matched_skills": [list(matched_skills)],
            "missing_skills": [list(missing_skills)],
            "cv_years": <cv_years>,
            "cv_job_title": <cv_job_title>
            "jd_years": <jd_years>,
            "review": <nhận xét chung, điểm cần cải thiện CV nếu có, bằng tiếng Việt>}
            """
        full_prompt = f"{prompt}\nJD: {jd}\n\n {cv} "

        # 2. Call generate_content
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=config,  # Pass the configuration here
        )

        # 3. Access the response text
        return json.loads(response.json())["parsed"]

    except Exception as e:
        print(traceback.format_exc())
    return ""


def match_job(cv_obj, jd_obj):
    content_cv = extract_text(cv_obj)
    content_jd = extract_text(jd_obj)
    analysis = llm_match_job(content_cv, content_jd)
    suggestions = []

    if analysis["overall_score"] < 50:
        suggestions.append(
            "🔴 **Độ phù hợp thấp**: CV cần chỉnh sửa đáng kể để phù hợp với JD này."
        )
    elif analysis["overall_score"] < 70:
        suggestions.append(
            "🟡 **Độ phù hợp trung bình**: CV có tiềm năng, cần cải thiện một số điểm."
        )
    else:
        suggestions.append(
            "🟢 **Độ phù hợp cao**: CV khớp tốt với JD, chỉ cần tinh chỉnh nhỏ."
        )

    if analysis["missing_skills"]:
        top_missing = analysis["missing_skills"][:5]
        suggestions.append(f"📚 **Bổ sung kỹ năng**: {', '.join(top_missing)}")

    suggestions.append(f"📊 **Nhận xét chung:** {analysis['review']}")

    return "<br> <br>".join(suggestions).replace("\n", "<br>")


# def main(server_port: int=8000, server_name: str = "0.0.0.0", debug=True) -> None:
with gr.Blocks(
    title="🎯 IT Job Market Analysis", css=custom_css, fill_width=True
) as demo:
    df = load_data()
    with gr.Tab("🌍 Market Overview") as tab1:
        # HEADER
        gr.HTML(
            """
                <div class="gradient-header">
                    <h1>🎯 IT Job Market Overview</h1>
                    <p>Market-wide insights on salary, locations, industries & companies</p>
                </div>
                """
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    """
                            <div class="filter-box">
                                <h3>🔍 Market Filters</h3>
                            </div>
                            """
                )

                industry_filter = gr.Dropdown(
                    label="📌 Industry",
                    choices=["All"] + sorted(df["industry"].unique().tolist()),
                )

                location_filter = gr.Dropdown(
                    label="📍 Location",
                    choices=["All"] + sorted(df["main_location"].unique().tolist()),
                )

            with gr.Column(scale=NUM_SCALE):
                gr.Markdown("## 📊 Key Metrics")

                with gr.Row():
                    total_jobs = gr.Number(
                        label="📌 Total Jobs", interactive=False, value=len(df)
                    )
                    companies = gr.Number(
                        label="🏢 Companies",
                        interactive=False,
                        value=df["company"].nunique(),
                    )
                    avg_min_salary = gr.Textbox(
                        label="💰 Avg Min Salary",
                        interactive=False,
                        value=cal_mean_format(df, "salary_min"),
                    )
                    avg_max_salary = gr.Textbox(
                        label="💰 Avg Max Salary",
                        interactive=False,
                        value=cal_mean_format(df, "salary_max"),
                    )

                gr.Markdown("---")

                # --------------------------
                # CHARTS SECTION
                # --------------------------
                gr.Markdown("## 🌍 Geographic & Industry Overview")

                with gr.Row():
                    chart_location = gr.Plot(create_jobs_by_location(df))
                    chart_industry = gr.Plot(create_industry_distribution(df))

                gr.Markdown("## 📈 Experience & Salary Analysis")

                with gr.Row():
                    chart_exp = gr.Plot(create_experience_chart(df))
                    chart_salary = gr.Plot(create_salary_distribution(df))

                gr.Markdown("## 💼 Salary & Company Analysis")

                with gr.Row():
                    chart_sal_industry = gr.Plot(create_salary_by_industry(df))
                    chart_top_company = gr.Plot(create_top_companies(df))

                gr.Markdown("## 🔧 Skills Overview")
                wordcloud_plot = gr.Plot(create_wordcloud_compact(df))

                # --------------------------
                # DATA TABLE
                # --------------------------
                gr.Markdown("## 📋 Job Listings")
                df_table = gr.Dataframe(
                    create_job_list_table(df),
                    label="Job Results",
                    interactive=False,
                    column_widths=COL_WIDTHS,
                )

        tab1.select(
            fn=apply_filters,
            inputs=[industry_filter, location_filter],
            outputs=[
                total_jobs,
                companies,
                avg_min_salary,
                avg_max_salary,
                chart_location,
                chart_industry,
                chart_exp,
                chart_salary,
                chart_sal_industry,
                chart_top_company,
                wordcloud_plot,
                df_table,
            ],
        )
        # Connect filters → update outputs
        industry_filter.change(
            fn=apply_filters,
            inputs=[industry_filter, location_filter],
            outputs=[
                total_jobs,
                companies,
                avg_min_salary,
                avg_max_salary,
                chart_location,
                chart_industry,
                chart_exp,
                chart_salary,
                chart_sal_industry,
                chart_top_company,
                wordcloud_plot,
                df_table,
            ],
        )

        location_filter.change(
            fn=apply_filters,
            inputs=[industry_filter, location_filter],
            outputs=[
                total_jobs,
                companies,
                avg_min_salary,
                avg_max_salary,
                chart_location,
                chart_industry,
                chart_exp,
                chart_salary,
                chart_sal_industry,
                chart_top_company,
                wordcloud_plot,
                df_table,
            ],
        )

    with gr.Tab("🔧 Skills Analysis") as tab2:
        # HEADER
        gr.HTML(
            """
            <div class="gradient-header">
                <h1>🔧 Skills Analysis by Industry</h1>
                <p>Discover in-demand skills for each industry segment</p>
            </div>
            """
        )
        # ROW LAYOUT: SIDEBAR (1) + MAIN PANEL (3)
        with gr.Row():

            # ----------------- SIDEBAR -----------------
            with gr.Column(scale=1):

                gr.HTML(
                    """
                    <div class="filter-box">
                        <h3>🔍 Skills Filters</h3>
                    </div>
                    """
                )

                industry_filter = gr.Dropdown(
                    label="📌 Select Industry",
                    choices=sorted(df["industry"].unique().tolist()),
                    value=sorted(df["industry"].unique().tolist())[0],
                )

                location_filter = gr.Dropdown(
                    label="📍 Select Locations",
                    choices=sorted(df["main_location"].unique().tolist()),
                    multiselect=True,
                )
                time_start_filter = gr.DateTime(
                    label="Select Time Start",
                    type="datetime",  # Ensures both date and time selection
                    value=datetime.datetime.now().timestamp() - 7 * 24 * 60 * 60,
                )
                time_end_filter = gr.DateTime(
                    label="Select Time End",
                    type="datetime",  # Ensures both date and time selection
                    value=datetime.datetime.now(),
                )

                btn_time_filter = gr.Button(value="Apply time")

            # ----------------- MAIN PANEL -----------------
            with gr.Column(scale=NUM_SCALE):

                # METRICS
                gr.Markdown("## 📊 Industry Overview")
                with gr.Row():
                    industry_title = gr.Textbox(label="Industry", interactive=False)
                    jobs = gr.Number(
                        label="📌 Jobs",
                        interactive=False,
                    )
                    companies = gr.Number(label="🏢 Companies", interactive=False)
                    avg_salary = gr.Textbox(label="💰 Avg Salary", interactive=False)
                    top_loc = gr.Textbox(label="📍 Top Location", interactive=False)
                    top_exp = gr.Textbox(label="💼 Common Exp.", interactive=False)

                gr.Markdown("---")

                # SKILLS
                gr.Markdown("## 🔧 Top Skills for This Industry")
                with gr.Row():
                    chart_skills = gr.Plot()
                    table_skills = gr.Dataframe(
                        interactive=False, wrap=True, max_height=HEIGHT
                    )

                gr.Markdown("### 🌟 Skills Word Cloud")
                chart_wc = gr.Plot()

                # JOB TITLES
                gr.Markdown("## 💼 Popular Job Titles")
                with gr.Row():
                    chart_titles = gr.Plot()
                    chart_location = gr.Plot()

                # SALARY
                gr.Markdown("## 💰 Salary Analysis")
                with gr.Row():
                    chart_salary_title = gr.Plot()
                    chart_salary_dist = gr.Plot()

                # TABLE
                gr.Markdown("## 📋 Job Listings")
                table_jobs = gr.Dataframe(
                    interactive=False, wrap=True, column_widths=COL_WIDTHS
                )

        tab2.select(
            fn=apply_filters_analysis,
            inputs=[
                industry_filter,
                location_filter,
                time_start_filter,
                time_end_filter,
            ],
            outputs=[
                industry_title,
                jobs,
                companies,
                avg_salary,
                top_loc,
                top_exp,
                chart_skills,
                table_skills,
                chart_wc,
                chart_titles,
                chart_location,
                chart_salary_title,
                chart_salary_dist,
                table_jobs,
            ],
        )
        # CONNECT FILTER EVENTS
        for widget in (industry_filter, location_filter):
            widget.change(
                fn=apply_filters_analysis,
                inputs=[
                    industry_filter,
                    location_filter,
                    time_start_filter,
                    time_end_filter,
                ],
                outputs=[
                    industry_title,
                    jobs,
                    companies,
                    avg_salary,
                    top_loc,
                    top_exp,
                    chart_skills,
                    table_skills,
                    chart_wc,
                    chart_titles,
                    chart_location,
                    chart_salary_title,
                    chart_salary_dist,
                    table_jobs,
                ],
            )
        btn_time_filter.click(
            fn=apply_filters_analysis,
            inputs=[
                industry_filter,
                location_filter,
                time_start_filter,
                time_end_filter,
            ],
            outputs=[
                industry_title,
                jobs,
                companies,
                avg_salary,
                top_loc,
                top_exp,
                chart_skills,
                table_skills,
                chart_wc,
                chart_titles,
                chart_location,
                chart_salary_title,
                chart_salary_dist,
                table_jobs,
            ],
        )
    with gr.Tab("📄 Job Matching") as tab3:
        with gr.Row():
            cv_file = gr.File(
                label="Upload your CV (PDF, Images, Docx, ...)",
                file_count="single",  # Options: "single" (default), "multiple", "directory"
                type="filepath",  # Returns the path to the temporary file
                file_types=["image", ".pdf", "docx"],
            )
            jd_file = gr.File(
                label="Upload the JD (PDF, Images, Docx, ...)",
                file_count="single",
                type="filepath",
                file_types=["image", ".pdf", "docx"],
            )
        btn_analyzer = gr.Button(value="Analyze")
        output_analyzer = gr.Markdown(label="Result")
        btn_analyzer.click(
            fn=match_job, inputs=[cv_file, jd_file], outputs=[output_analyzer]
        )
server_name = "0.0.0.0"
server_port = 8001

demo.launch(debug=True, server_name=server_name, server_port=server_port, share=True)

# if __name__=="__main__":
#     main()
