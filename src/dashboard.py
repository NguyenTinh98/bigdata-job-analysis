#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IT Job Market Dashboard - Multi-Page Streamlit Version
Page 1: Market Overview (Salary ranges fixed, compact wordcloud)
Page 2: Skills Analysis by Industry (Industry filter, gradient theme)
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import psycopg2

# === CONFIGURATION ===
DB_NAME = "job_database"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

TABLE_NAME = "topcv_jobs"  # name for new table

conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
cursor = conn.cursor()


# Auto-install dependencies
def install_dependencies():
    """Automatically install required packages"""
    packages = [
        "streamlit",
        "plotly",
        "pandas",
        "openpyxl",
        "wordcloud",
        "matplotlib",
        "pandas",
    ]

    for package in packages:
        try:
            __import__(package if package != "openpyxl" else "openpyxl")
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"]
            )


install_dependencies()

import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
from sqlalchemy import create_engine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="IT Job Market Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM THEME - GRADIENT COLORS
# ============================================================================

PRIMARY_GRADIENT = "#667eea"
SECONDARY_GRADIENT = "#764ba2"
ACCENT_COLOR = "#FF6B6B"

# Custom CSS with gradient theme
st.markdown(
    f"""
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
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}

    .stMetric {{
        background-color: rgba(102, 126, 234, 0.05);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid {PRIMARY_GRADIENT};
    }}

    .section-title {{
        background: linear-gradient(90deg, {PRIMARY_GRADIENT} 0%, {SECONDARY_GRADIENT} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .filter-box {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid {PRIMARY_GRADIENT};
        margin-bottom: 10px;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================


@st.cache_data(ttl=0, show_spinner=False)
def load_data():
    """Load and process job data from Excel file"""
    # try:
    #     df = pd.read_excel(PATH)
    # except FileNotFoundError:
    #     st.error(f"❌ File {PATH} not found!")
    #     st.stop()
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME};", engine)
    engine.dispose()

    return df


@st.cache_data(ttl=0, show_spinner=False)
def process_data(df):
    """Process and extract data from raw dataframe"""

    # Extract skills from text
    def extract_skills_from_text(text):
        if pd.isna(text):
            return []

        text = str(text).lower()
        skills = []

        tech_skills = {
            "python",
            "java",
            "javascript",
            "js",
            "typescript",
            "c++",
            "c#",
            "php",
            "ruby",
            "go",
            "golang",
            "swift",
            "kotlin",
            "r",
            "scala",
            "perl",
            "html",
            "css",
            "react",
            "reactjs",
            "angular",
            "angularjs",
            "vue",
            "vuejs",
            "nodejs",
            "node.js",
            "express",
            "django",
            "flask",
            "spring",
            "laravel",
            "jquery",
            "bootstrap",
            "tailwind",
            "sql",
            "mysql",
            "postgresql",
            "mongodb",
            "oracle",
            "sql server",
            "redis",
            "elasticsearch",
            "cassandra",
            "dynamodb",
            "aws",
            "azure",
            "gcp",
            "google cloud",
            "docker",
            "kubernetes",
            "jenkins",
            "gitlab",
            "github",
            "ci/cd",
            "terraform",
            "ansible",
            "machine learning",
            "deep learning",
            "ai",
            "data science",
            "big data",
            "tensorflow",
            "pytorch",
            "pandas",
            "numpy",
            "scikit-learn",
            "git",
            "jira",
            "confluence",
            "figma",
            "sketch",
            "adobe",
            "photoshop",
            "excel",
            "power bi",
            "tableau",
            "agile",
            "scrum",
            "rest api",
            "graphql",
            "microservices",
            "linux",
            "windows",
            "macos",
        }

        for skill in tech_skills:
            if skill in text:
                skills.append(skill.title())

        return list(set(skills))

    df["skills_desc"] = df["desc_mota"].apply(extract_skills_from_text)
    df["skills_req"] = df["desc_yeucau"].apply(extract_skills_from_text)
    df["skills_title"] = df["title"].apply(extract_skills_from_text)

    def combine_skills(row):
        all_skills = []
        all_skills.extend(row["skills_desc"])
        all_skills.extend(row["skills_req"])
        all_skills.extend(row["skills_title"])
        return list(set(all_skills))

    df["all_skills"] = df.apply(combine_skills, axis=1)

    # Extract industry
    def extract_industry(tags_text, title_text):
        if pd.isna(tags_text):
            tags_text = ""
        if pd.isna(title_text):
            title_text = ""

        text = str(tags_text).lower() + " " + str(title_text).lower()

        if "it - phần mềm" in text or "software" in text:
            return "IT - Software"
        elif "backend" in text or "frontend" in text or "fullstack" in text:
            return "Software Development"
        elif "data" in text or "ai" in text or "machine learning" in text:
            return "Data & AI"
        elif "devops" in text or "cloud" in text:
            return "DevOps & Cloud"
        elif "project manager" in text or "product manager" in text:
            return "Product & Project Management"
        elif (
            "thiết kế" in text
            or "design" in text
            or "graphic" in text
            or "ui/ux" in text
        ):
            return "Design"
        elif "kinh doanh" in text or "sales" in text or "business" in text:
            return "Business & Sales"
        elif "test" in text or "qa" in text or "quality" in text:
            return "QA & Testing"
        elif "security" in text or "bảo mật" in text:
            return "Security"
        elif "mobile" in text or "android" in text or "ios" in text:
            return "Mobile Development"
        elif "game" in text:
            return "Gaming"
        elif "marketing" in text:
            return "Marketing"
        else:
            return "Other"

    df["industry_final"] = df.apply(
        lambda row: extract_industry(row["tags"], row["title"]), axis=1
    )

    # Parse salary
    def parse_salary_range(salary_text):
        if (
            pd.isna(salary_text)
            or "Thoả thuận" in str(salary_text)
            or "thuận" in str(salary_text).lower()
        ):
            return None, None

        text = str(salary_text)
        numbers = re.findall(r"(\d+)", text)

        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        elif len(numbers) == 1:
            if "Trên" in text or "trên" in text:
                return int(numbers[0]), None
            elif "Tới" in text or "Đến" in text or "đến" in text:
                return None, int(numbers[0])
            else:
                return int(numbers[0]), int(numbers[0])

        return None, None

    df[["salary_min", "salary_max"]] = df["salary_list"].apply(
        lambda x: pd.Series(parse_salary_range(x))
    )

    df["salary_avg"] = df.apply(
        lambda row: (
            (row["salary_min"] + row["salary_max"]) / 2
            if pd.notna(row["salary_min"]) and pd.notna(row["salary_max"])
            else (
                row["salary_min"]
                if pd.notna(row["salary_min"])
                else row["salary_max"] if pd.notna(row["salary_max"]) else None
            )
        ),
        axis=1,
    )

    # Extract location
    def extract_main_location(location_text):
        if pd.isna(location_text):
            return "Unknown"

        text = str(location_text)
        if "Hà Nội" in text:
            return "Hà Nội"
        elif "Hồ Chí Minh" in text or "HCM" in text:
            return "Hồ Chí Minh"
        elif "Đà Nẵng" in text:
            return "Đà Nẵng"
        elif "Bình Dương" in text:
            return "Bình Dương"
        elif "Đồng Nai" in text:
            return "Đồng Nai"
        elif "Hải Phòng" in text:
            return "Hải Phòng"
        else:
            return text.split(",")[0].strip()

    df["main_location"] = df["address_list"].apply(extract_main_location)

    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def create_salary_by_industry(filtered_df):
    """Average salary by industry - FIXED HEIGHT"""
    if filtered_df.empty or filtered_df["salary_avg"].notna().sum() == 0:
        return None

    salary_data = (
        filtered_df.groupby("industry_final")["salary_avg"]
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
    fig.update_layout(height=350, showlegend=False)
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
        color=location_data.values,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
    return fig


def create_industry_distribution(filtered_df):
    """Industry distribution"""
    if filtered_df.empty:
        return None

    industry_data = filtered_df["industry_final"].value_counts()

    fig = px.pie(
        values=industry_data.values,
        names=industry_data.index,
        title="🏭 Job Distribution by Industry",
        hole=0.4,
    )
    fig.update_layout(height=400)
    return fig


def create_experience_chart(filtered_df):
    """Experience level distribution"""
    if filtered_df.empty:
        return None

    exp_data = filtered_df["exp_list"].value_counts()

    fig = px.bar(
        x=exp_data.index,
        y=exp_data.values,
        title="💼 Jobs by Experience Level",
        labels={"x": "Experience", "y": "Number of Jobs"},
        color=exp_data.values,
        color_continuous_scale="Greens",
    )
    fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
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
    fig.update_layout(height=350, showlegend=False)
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
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_wordcloud_compact(filtered_df, height=300, width=600):
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


def page_market_overview():
    """Market Overview Dashboard"""

    # Header
    st.markdown(
        """
        <div class="gradient-header">
            <h1>🎯 IT Job Market Overview</h1>
            <p>Market-wide insights on salary, locations, industries & companies</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Load data
    df = load_data()
    df = process_data(df)

    # Sidebar Filters
    st.sidebar.markdown(
        """
        <div class="filter-box">
            <h3>🔍 Market Filters</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    selected_industry = st.sidebar.selectbox(
        "📌 Industry", ["All"] + sorted(df["industry_final"].unique().tolist())
    )

    selected_location = st.sidebar.selectbox(
        "📍 Location", ["All"] + sorted(df["main_location"].unique().tolist())
    )

    # Apply filters
    filtered_df = df.copy()

    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["industry_final"] == selected_industry]

    if selected_location != "All":
        filtered_df = filtered_df[filtered_df["main_location"] == selected_location]

    # Key Metrics
    st.markdown("## 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📌 Total Jobs", len(filtered_df))

    with col2:
        st.metric("🏢 Companies", filtered_df["company"].nunique())

    with col3:
        avg_sal = (
            filtered_df["salary_avg"].mean()
            if filtered_df["salary_avg"].notna().sum() > 0
            else 0
        )
        st.metric("💰 Avg Salary", f"{avg_sal:.1f}M" if avg_sal > 0 else "N/A")

    with col4:
        salary_count = filtered_df["salary_avg"].notna().sum()
        st.metric("📊 Salary Info", salary_count)

    st.markdown("---")

    # Row 1: Location & Industry
    st.markdown("## 🌍 Geographic & Industry Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig = create_jobs_by_location(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_industry_distribution(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Row 2: Experience & Salary
    st.markdown("## 📈 Experience & Salary Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = create_experience_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_salary_distribution(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Row 3: Salary by Industry & Top Companies
    st.markdown("## 💼 Salary & Company Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = create_salary_by_industry(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_top_companies(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Row 4: Compact Wordcloud
    st.markdown("## 🔧 Skills Overview")
    fig = create_wordcloud_compact(filtered_df, height=250, width=1000)
    if fig:
        st.pyplot(fig, use_container_width=True)

    # Data Table
    st.markdown("## 📋 Job Listings")
    display_cols = [
        "title",
        "company",
        "main_location",
        "salary_avg",
        "exp_list",
        "industry_final",
    ]
    display_df = filtered_df[display_cols].copy()
    display_df.columns = [
        "Job Title",
        "Company",
        "Location",
        "Salary (M)",
        "Experience",
        "Industry",
    ]
    display_df["Salary (M)"] = display_df["Salary (M)"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE 2: SKILLS ANALYSIS BY INDUSTRY
# ============================================================================


def page_skills_analysis():
    """Skills Analysis Dashboard - Focused on Industry-specific Skills"""

    # Header
    st.markdown(
        """
        <div class="gradient-header">
            <h1>🔧 Skills Analysis by Industry</h1>
            <p>Discover in-demand skills for each industry segment</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Load data
    df = load_data()
    df = process_data(df)

    # Sidebar Filters
    st.sidebar.markdown(
        """
        <div class="filter-box">
            <h3>🔍 Skills Filters</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    selected_industry = st.sidebar.selectbox(
        "📌 Select Industry",
        sorted(df["industry_final"].unique().tolist()),
        key="skills_industry",
    )

    selected_location = st.sidebar.multiselect(
        "📍 Select Locations (optional)",
        sorted(df["main_location"].unique().tolist()),
        default=[],
        key="skills_location",
    )

    # Apply filters
    filtered_df = df[df["industry_final"] == selected_industry].copy()

    if selected_location:
        filtered_df = filtered_df[filtered_df["main_location"].isin(selected_location)]

    if filtered_df.empty:
        st.warning("No jobs found for selected filters")
        return

    # Industry Overview
    st.markdown("## 📊 Industry Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📌 Jobs", len(filtered_df))

    with col2:
        st.metric("🏢 Companies", filtered_df["company"].nunique())

    with col3:
        avg_sal = (
            filtered_df["salary_avg"].mean()
            if filtered_df["salary_avg"].notna().sum() > 0
            else 0
        )
        st.metric("💰 Avg Salary", f"{avg_sal:.1f}M" if avg_sal > 0 else "N/A")

    with col4:
        top_loc = (
            filtered_df["main_location"].value_counts().index[0]
            if not filtered_df.empty
            else "N/A"
        )
        st.metric("📍 Top Location", top_loc)

    with col5:
        top_exp = (
            filtered_df["exp_list"].value_counts().index[0]
            if not filtered_df.empty
            else "N/A"
        )
        st.metric("💼 Common Exp.", top_exp)

    st.markdown("---")

    # Skills Analysis
    st.markdown("## 🔧 Top Skills for This Industry")

    # Extract all skills
    all_skills = []
    for skills in filtered_df["all_skills"]:
        all_skills.extend(skills)

    if all_skills:
        skill_freq = Counter(all_skills)

        # Create skills visualization
        col1, col2 = st.columns(2)

        with col1:
            # Skills Bar Chart
            top_skills = dict(skill_freq.most_common(15))
            fig = px.bar(
                x=list(top_skills.values()),
                y=list(top_skills.keys()),
                orientation="h",
                title=f"Top 15 Skills - {selected_industry}",
                labels={"x": "Frequency", "y": "Skill"},
                color=list(top_skills.values()),
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Skills Table
            skills_df = pd.DataFrame(
                skill_freq.most_common(20), columns=["Skill", "Frequency"]
            )
            st.markdown(f"### 📋 Top 20 Skills")
            st.dataframe(skills_df, use_container_width=True, hide_index=True)

    # Word Cloud - Larger for skills focus
    st.markdown("### 🌟 Skills Word Cloud")
    fig = create_wordcloud_compact(filtered_df, height=400, width=1000)
    if fig:
        st.pyplot(fig, use_container_width=True)

    # Job Titles in this industry
    st.markdown("## 💼 Popular Job Titles")

    col1, col2 = st.columns(2)

    with col1:
        titles = filtered_df["title"].value_counts().head(15)
        fig = px.bar(
            x=titles.values,
            y=titles.index,
            orientation="h",
            title="Top Job Titles",
            labels={"x": "Count", "y": "Job Title"},
            color=titles.values,
            color_continuous_scale="Greens",
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Location distribution within industry
        locations = filtered_df["main_location"].value_counts()
        fig = px.pie(
            values=locations.values,
            names=locations.index,
            title="Location Distribution",
            hole=0.4,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Salary Analysis for industry
    st.markdown("## 💰 Salary Analysis")

    col1, col2 = st.columns(2)

    with col1:
        salary_by_title = (
            filtered_df.groupby("title")["salary_avg"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        if len(salary_by_title) > 0:
            fig = px.bar(
                x=salary_by_title.values,
                y=salary_by_title.index,
                orientation="h",
                title="Salary by Job Title",
                labels={"x": "Average Salary (M)", "y": "Job Title"},
                color=salary_by_title.values,
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_salary_distribution(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Detailed Job Listings
    st.markdown("## 📋 Job Listings for {selected_industry}")
    display_cols = ["title", "company", "main_location", "salary_avg", "exp_list"]
    display_df = filtered_df[display_cols].copy()
    display_df.columns = [
        "Job Title",
        "Company",
        "Location",
        "Salary (M)",
        "Experience",
    ]
    display_df["Salary (M)"] = display_df["Salary (M)"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP - PAGE NAVIGATION
# ============================================================================


def main():
    # Page Navigation
    page = st.sidebar.radio(
        "📑 Select Page", ["🌍 Market Overview", "🔧 Skills Analysis"], key="page_nav"
    )

    if page == "🌍 Market Overview":
        page_market_overview()
    elif page == "🔧 Skills Analysis":
        page_skills_analysis()


if __name__ == "__main__":
    main()
