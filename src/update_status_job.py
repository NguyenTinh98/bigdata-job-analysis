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

CONFIG = Variable.get("close_job", deserialize_json=True)

DB_NAME = CONFIG["DB_NAME"]
DB_USER = CONFIG["DB_USER"]
DB_PASSWORD = CONFIG["DB_PASSWORD"]
DB_HOST = CONFIG["DB_HOST"]
DB_PORT = CONFIG["DB_PORT"]
TABLE_NAME = CONFIG["TABLE_NAME"]
SCHEDULE = CONFIG["SCHEDULE"]
MAX_ACTIVATE_RUN = int(CONFIG["max_active_runs"])


@task
def close_job():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    cursor = conn.cursor()

    # === INSERT DATA ===
    query = f"""
    UPDATE {TABLE_NAME}
    SET status = 'closed'
    WHERE deadline < NOW() AND status != 'closed';
    """

    cursor.execute(query)
    conn.commit()
    updated_rows = cursor.rowcount
    print(f"Closed {updated_rows} jobs.")

    cursor.close()
    conn.close()


with DAG(
    dag_id="close_job",
    description="This DAG runs a workflow for job crawling from TOPCV",
    schedule=SCHEDULE,
    start_date=datetime(2021, 1, 1),
    max_active_runs=MAX_ACTIVATE_RUN,
    catchup=False,
    tags=["bigdata"],
) as dag:

    close_job()
