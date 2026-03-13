import os
import base64
import sqlite3
import uuid
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import numpy as np
import re
import matplotlib.pyplot as plt
import urllib.request
import urllib.parse
import urllib.error
from groq import Groq as GroqClient

# ── LangSmith tracing ─────────────────────────────────────────────
try:
    from langsmith import Client as LangSmithClient
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator

load_dotenv(override=True)

# ── Set LangSmith env vars explicitly from .env values ───────────
for _key in ("LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"):
    _val = os.getenv(_key)
    if _val:
        os.environ[_key] = _val

# ── LangSmith client (initialised once) ──────────────────────────
def _get_langsmith_client():
    if not _LANGSMITH_AVAILABLE:
        return None
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return None
    try:
        return LangSmithClient(api_key=api_key)
    except Exception:
        return None

_ls_client = _get_langsmith_client()

def _ls_enabled() -> bool:
    return (
        _LANGSMITH_AVAILABLE
        and bool(os.getenv("LANGCHAIN_API_KEY"))
        and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )


# =========================================================
# FREE ALERT SYSTEM (ntfy.sh — no signup, no cost)
# =========================================================
ALERT_THRESHOLDS = {
    "latency_warning_sec":  8.0,
    "latency_critical_sec": 20.0,
    "error_burst_limit":    3,
}

def _send_alert(title: str, message: str, priority: str = "default", tags: str = "bell") -> None:
    topic = os.getenv("NTFY_TOPIC", "")
    if not topic:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title":    title,
                "Priority": priority,
                "Tags":     tags,
            },
            timeout=4,
        )
    except Exception:
        pass


def _alert_rate_limit(feature_tag: str) -> None:
    _send_alert(
        title    = "Alert - Rate Limit Hit",
        message  = f"What: Groq API rate limit exceeded\nWhere: {feature_tag}\nFix: Wait 60 seconds and retry",
        priority = "high",
        tags     = "warning",
    )


def _alert_error(feature_tag: str, error_type: str, error_msg: str) -> None:
    _send_alert(
        title    = "Alert - LLM Error",
        message  = f"What: {error_type}\nWhere: {feature_tag}\nDetails: {error_msg[:200]}",
        priority = "urgent",
        tags     = "rotating_light",
    )


def _alert_slow_response(feature_tag: str, latency: float) -> None:
    if latency >= ALERT_THRESHOLDS["latency_critical_sec"]:
        _send_alert(
            title    = "Alert - Very Slow Response",
            message  = f"What: Response took {latency:.1f} seconds (critical)\nWhere: {feature_tag}\nThreshold: {ALERT_THRESHOLDS['latency_critical_sec']}s",
            priority = "high",
            tags     = "rotating_light",
        )
    elif latency >= ALERT_THRESHOLDS["latency_warning_sec"]:
        _send_alert(
            title    = "Alert - Slow Response",
            message  = f"What: Response took {latency:.1f} seconds\nWhere: {feature_tag}\nThreshold: {ALERT_THRESHOLDS['latency_warning_sec']}s",
            priority = "default",
            tags     = "warning",
        )


def _alert_adzuna_down() -> None:
    _send_alert(
        title    = "Alert - Adzuna API Down",
        message  = "What: Job listings API is not responding\nWhere: Job search / Market Insights\nFix: Adzuna may be temporarily down — try again later",
        priority = "high",
        tags     = "rotating_light,briefcase",
    )


def _alert_success_milestone(feature_tag: str) -> None:
    _send_alert(
        title    = "Alert - Feature Used",
        message  = f"What: A user completed an action\nWhere: {feature_tag}",
        priority = "min",
        tags     = "white_check_mark",
    )


def _increment_error_count(feature_tag: str) -> int:
    key = "_alert_error_count"
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] += 1
    count = st.session_state[key]
    if count >= ALERT_THRESHOLDS["error_burst_limit"]:
        _send_alert(
            title    = "Alert - Repeated Errors",
            message  = f"What: {count} LLM errors in a single session\nWhere: {feature_tag}\nFix: Check your Groq API key and rate limits",
            priority = "urgent",
            tags     = "rotating_light,sos",
        )
        st.session_state[key] = 0
    return count

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SMART CAREERFORDGE",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOGO HELPER
# =========================================================
def _get_logo_b64():
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

_LOGO_B64 = _get_logo_b64()

# =========================================================
# GLOBAL STYLES
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #000000;
        color: #E6E6E6;
    }
    .stApp { background: #000000 !important; }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }
    p, li, td, th, div, span,
    .stMarkdown p, .stMarkdown li,
    .stWrite, .stText { color: #E6E6E6 !important; }
    pre, code, .stMarkdown pre, .stMarkdown code { color: #E5E7EB !important; line-height: 1.6 !important; }
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-family: 'Syne', sans-serif !important; color: #F3F4F6 !important; font-weight: 600 !important;
    }
    strong, b { color: #F3F4F6 !important; font-weight: 600 !important; }
    label, .stFileUploader label, .stSelectbox label, .stRadio label,
    .stTextInput label, [data-testid="stWidgetLabel"] { color: #F3F4F6 !important; font-weight: 600 !important; }
    .screen-subheader, .stCaption, small, caption,
    .stFileUploader small,
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: #9CA3AF !important; }

    /* BUTTON STYLES */
    .stButton > button {
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 5px 10px !important;
        min-height: 32px !important;
        height: auto !important;
        line-height: 1.3 !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.18s ease !important;
    }
    .stButton > button *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        color: #000000 !important;
        background: transparent !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .stButton > button:hover,
    .stButton > button:hover *,
    .stButton > button:hover span,
    .stButton > button:hover p {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }
    .stButton > button[kind="primary"] {
        background: #22D3EE !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        padding: 7px 18px !important;
        white-space: nowrap !important;
    }
    .stButton > button[kind="primary"] * { color: #000000 !important; background: transparent !important; }
    .stButton > button[kind="primary"]:hover { background: #2563EB !important; color: #FFFFFF !important; }
    .stButton > button[kind="primary"]:hover * { color: #FFFFFF !important; }
    .stDownloadButton > button {
        background: #FFFFFF !important; color: #000000 !important;
        border: 1px solid #E5E7EB !important; border-radius: 10px !important;
        font-weight: 600 !important; white-space: nowrap !important;
    }
    .stDownloadButton > button:hover { background: #2563EB !important; color: #FFFFFF !important; border-color: #2563EB !important; }

    .hero-title {
        font-family: 'Syne', sans-serif; font-size: 56px; font-weight: 800;
        letter-spacing: 1px; margin-bottom: 6px;
        background: linear-gradient(180deg,#F8FAFC 0%,#D1D5DB 30%,#9CA3AF 55%,#6B7280 75%,#E5E7EB 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        filter: drop-shadow(0 1px 1px rgba(255,255,255,0.25)) drop-shadow(0 4px 12px rgba(0,0,0,0.8)) drop-shadow(0 0 20px rgba(209,213,219,0.25));
    }
    .hero-sub { font-size: 0.95rem; color: #9CA3AF !important; margin-bottom: 36px; }
    .screen-header { font-family: 'Syne', sans-serif; font-size: 1.75rem; font-weight: 700; color: #F3F4F6 !important; margin-bottom: 4px; }

    .stTabs [data-baseweb="tab-list"] { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; border-radius: 12px !important; gap: 24px !important; padding: 6px 16px !important; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #CBD5E1 !important; font-family: 'Syne', sans-serif !important; font-size: 0.85rem !important; font-weight: 500 !important; background: transparent !important; padding: 10px 18px !important; margin: 0 4px !important; letter-spacing: 0.02em !important; }
    .stTabs [aria-selected="true"] { background: transparent !important; color: #22D3EE !important; font-weight: 700 !important; border-bottom: 2px solid #22D3EE !important; border-radius: 0px !important; padding: 10px 18px !important; }
    [data-testid="metric-container"] { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; border-radius: 12px !important; padding: 14px 18px !important; }
    [data-testid="metric-container"] label { color: #9CA3AF !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #F3F4F6 !important; font-weight: 700 !important; }
    .stTextInput > div > div > input { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; border-radius: 8px !important; color: #E6E6E6 !important; }
    .stTextInput > div > div > input::placeholder { color: #4B5563 !important; }
    .stSelectbox > div > div { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; color: #E6E6E6 !important; }
    .stSelectbox > div > div > div { color: #E6E6E6 !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] *, [data-baseweb="menu"], [data-baseweb="menu"] *, [role="listbox"], [role="listbox"] * { background: #FFFFFF !important; color: #000000 !important; }
    [role="option"], [role="option"] *, [role="option"] span, [role="option"] div, [role="option"] p, li[role="option"], li[role="option"] * { color: #000000 !important; background: #FFFFFF !important; font-weight: 500 !important; }
    [role="option"]:hover, [role="option"]:hover *, [role="option"][aria-selected="true"], [role="option"][aria-selected="true"] * { background: #DBEAFE !important; color: #000000 !important; }
    .stRadio > div { color: #E6E6E6 !important; }
    .streamlit-expanderHeader { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; border-radius: 10px !important; color: #CBD5E1 !important; }
    .streamlit-expanderContent { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #22D3EE, #2563EB) !important; border-radius: 4px !important; }
    [data-testid="stFileUploader"] { background: #0a0a0a !important; border: 1px dashed #2a2a2a !important; border-radius: 14px !important; padding: 8px !important; }
    [data-testid="stFileUploader"] button, [data-testid="stFileUploaderDropzone"] button, [data-testid="baseButton-secondary"] { background: #FFFFFF !important; color: #000000 !important; border: 1px solid #d1d5db !important; border-radius: 8px !important; font-weight: 600 !important; }
    [data-testid="stFileUploader"] button:hover, [data-testid="stFileUploaderDropzone"] button:hover { background: #2563EB !important; color: #FFFFFF !important; border-color: #2563EB !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span { color: #000000 !important; font-weight: 600 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small { color: #6B7280 !important; }
    [data-testid="stAlert"][data-type="success"] { background: #052E16 !important; color: #D1FAE5 !important; border-radius: 10px !important; }
    .stSuccess { background: #052E16 !important; color: #D1FAE5 !important; border-radius: 10px !important; }
    .stInfo { border-radius: 10px !important; }
    .stWarning { border-radius: 10px !important; }
    hr { border-color: #1f1f1f !important; }
    .stSpinner > div { border-top-color: #22D3EE !important; }
    .stTextArea textarea { background: #0a0a0a !important; border: 1px solid #1f1f1f !important; border-radius: 8px !important; color: #E6E6E6 !important; }

    /* SPLASH */
    .splash-logo { display: block; margin: 60px auto 0 auto; width: 100%; max-width: 720px; border-radius: 18px; box-shadow: 0 0 80px rgba(220,220,220,0.12), 0 0 200px rgba(255,255,255,0.05); animation: splashFade 1.2s cubic-bezier(0.16,1,0.3,1) both; }
    .splash-tagline { text-align: center; margin-top: 20px; margin-bottom: 24px; font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: #9CA3AF; letter-spacing: 0.05em; animation: splashFade 1.2s 0.3s cubic-bezier(0.16,1,0.3,1) both; }
    @keyframes splashFade { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }

    /* CHAT UI */
    .msg-user { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 18px 18px 4px 18px; padding: 14px 18px; margin: 6px 0 6px 18%; color: #E6E6E6 !important; font-size: 0.9rem; line-height: 1.65; }
    .msg-assistant { background: #080f18; border: 1px solid #1a2535; border-left: 3px solid #22D3EE; border-radius: 0 18px 18px 18px; padding: 14px 18px; margin: 6px 18% 6px 0; color: #E6E6E6 !important; font-size: 0.9rem; line-height: 1.65; }
    .msg-label-user { text-align: right; font-size: 0.7rem; color: #22D3EE !important; margin-bottom: 2px; padding-right: 6px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
    .msg-label-ai { text-align: left; font-size: 0.7rem; color: #6B7280 !important; margin-bottom: 2px; padding-left: 6px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
    .status-bar { background: #050c14; border: 1px solid #0e1e2e; border-radius: 10px; padding: 9px 16px; margin-bottom: 16px; font-size: 0.78rem; color: #4B5563 !important; }
    .green-dot { width: 7px; height: 7px; border-radius: 50%; background: #22D3EE; box-shadow: 0 0 6px rgba(34,211,238,0.7); display: inline-block; margin-right: 8px; vertical-align: middle; }
    .resume-banner { background: #030d08; border: 1px solid #0a2e18; border-left: 3px solid #22D3EE; border-radius: 8px; padding: 9px 14px; margin-bottom: 14px; font-size: 0.79rem; color: #6ee7b7 !important; }
    .section-hint { font-size: 0.78rem; color: #4B5563 !important; margin-bottom: 8px; margin-top: 4px; }

    /* SCORE CARD */
    .score-card-a { background: linear-gradient(135deg,#052e16,#064e3b); border: 1px solid #059669; border-radius: 16px; padding: 24px; text-align: center; }
    .score-card-b { background: linear-gradient(135deg,#1e3a5f,#1e40af); border: 1px solid #3b82f6; border-radius: 16px; padding: 24px; text-align: center; }
    .score-card-c { background: linear-gradient(135deg,#78350f,#92400e); border: 1px solid #f59e0b; border-radius: 16px; padding: 24px; text-align: center; }
    .score-card-d { background: linear-gradient(135deg,#450a0a,#7f1d1d); border: 1px solid #ef4444; border-radius: 16px; padding: 24px; text-align: center; }
    .score-number { font-family: 'Syne',sans-serif; font-size: 4rem; font-weight: 800; color: #F3F4F6 !important; line-height: 1; }
    .score-grade { font-family: 'Syne',sans-serif; font-size: 2rem; font-weight: 700; margin-top: 4px; }
    .score-label { font-size: 0.82rem; color: #9CA3AF !important; margin-top: 6px; }

    /* JOB MATCH */
    .job-card { background: #080f18; border: 1px solid #1a2535; border-left: 3px solid #22D3EE; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; }
    .job-card-applied { background: #050c0f; border: 1px solid #1a2535; border-left: 3px solid #16a34a; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; opacity: 0.6; }
    .job-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #F3F4F6 !important; margin-bottom: 4px; }
    .job-meta { font-size: 0.78rem; color: #9CA3AF !important; margin-bottom: 8px; }
    .job-salary { font-size: 0.82rem; font-weight: 600; color: #22D3EE !important; margin-bottom: 6px; }
    .job-desc { font-size: 0.8rem; color: #6B7280 !important; line-height: 1.5; margin-bottom: 10px; }
    .job-exp-tag { display:inline-block; background:#0e2a35; color:#22D3EE !important; font-size:0.7rem; font-weight:600; padding:2px 8px; border-radius:10px; margin-bottom:8px; }
    .job-apply { display: inline-block; background: #FFFFFF; color: #000000 !important; font-size: 0.75rem; font-weight: 700; padding: 5px 14px; border-radius: 6px; text-decoration: none; }
    .job-apply:hover { background: #22D3EE; }
    .applied-badge { display:inline-block; background:#16a34a; color:#fff !important; font-size:0.7rem; font-weight:700; padding:3px 10px; border-radius:6px; margin-left:8px; }
    .match-pill { display:inline-block; font-size:0.72rem; font-weight:700; padding:2px 10px; border-radius:10px; margin-left:8px; }
    .match-high { background:#052e16; color:#22c55e !important; border:1px solid #16a34a; }
    .match-mid  { background:#1e3a5f; color:#60a5fa !important; border:1px solid #3b82f6; }
    .match-low  { background:#450a0a; color:#fca5a5 !important; border:1px solid #ef4444; }

    /* INSIGHT CARD */
    .insight-card { background:#080f18; border:1px solid #1a2535; border-radius:12px; padding:16px 18px; margin-bottom:10px; }
    .insight-number { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#22D3EE !important; }
    .insight-label { font-size:0.78rem; color:#9CA3AF !important; margin-top:2px; }

    /* FEEDBACK & RATINGS */
    .fb-card { background:#080f18; border:1px solid #1a2535; border-radius:14px; padding:18px 20px; margin-bottom:12px; transition: border-color 0.2s; }
    .fb-card:hover { border-color:#22D3EE; }
    .fb-author { font-family:'Syne',sans-serif; font-size:0.88rem; font-weight:700; color:#F3F4F6 !important; }
    .fb-feature { display:inline-block; background:#0e2a35; color:#22D3EE !important; font-size:0.68rem; font-weight:700; padding:2px 9px; border-radius:10px; margin-left:8px; vertical-align:middle; }
    .fb-date { font-size:0.7rem; color:#4B5563 !important; float:right; }
    .fb-comment { font-size:0.86rem; color:#CBD5E1 !important; line-height:1.65; margin-top:10px; }
    .fb-stars { font-size:1.1rem; margin-top:6px; }
    .stars-lg { font-size:2.8rem; letter-spacing:4px; }
    .rating-bar-wrap { background:#111; border-radius:6px; height:10px; width:100%; margin:4px 0; overflow:hidden; }
    .rating-bar-fill { background:linear-gradient(90deg,#22D3EE,#2563EB); height:10px; border-radius:6px; }
    .avg-score { font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800; color:#22D3EE !important; line-height:1; }
    .avg-label { font-size:0.78rem; color:#9CA3AF !important; margin-top:4px; }
    .hero-stat { background:#080f18; border:1px solid #1a2535; border-radius:12px; padding:16px; text-align:center; }
    .hero-stat-num { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#22D3EE !important; }
    .hero-stat-lbl { font-size:0.72rem; color:#9CA3AF !important; margin-top:2px; }
    .like-badge { display:inline-block; background:#052e16; color:#22c55e !important; font-size:0.7rem; font-weight:700; padding:2px 9px; border-radius:10px; border:1px solid #16a34a; cursor:pointer; }
    .tag-pill { display:inline-block; background:#1a1a2e; color:#818cf8 !important; font-size:0.68rem; font-weight:600; padding:2px 8px; border-radius:8px; margin-right:4px; border:1px solid #312e81; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE INIT
# =========================================================
def _init_state():
    defaults = {
        "current_screen": "splash",
        "resume_text": "",
        "resume_skills": "",
        "jd_text": "",
        "ats_result": None,
        "candidate_name": "",
        "assistant_messages": [],
        "assistant_input_key": 0,
        "job_search_role": "",
        "job_search_location": "india",
        "job_results": [],
        "applied_jobs": [],
        "applied_job_urls": [],
        "job_current_page": 1,
        "user_experience_level": "",
        "user_experience_years": 0,
        "exp_profile_done": False,
        "resume_score_card": None,
        "cover_letter": "",
        "salary_estimate": "",
        "linkedin_bio": "",
        "market_insights": {},
        "salary_chart_data": {},
        "feedback_submitted": False,
        "feedback_filter_feature": "All",
        "feedback_sort": "newest",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# =========================================================
# GROQ CLIENT — reads GROQ_API_KEY from .env
# =========================================================
GROQ_MODEL = "llama-3.3-70b-versatile"

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()
    return GroqClient(api_key=api_key)


@traceable(name="groq_llm_call", run_type="llm",
           metadata={"platform": "Smart CareerFordge", "model": "llama-3.3-70b-versatile"})
def _call_groq_direct(messages: list, system_prompt: str,
                       max_tokens: int = 1024, temperature: float = 0.3,
                       feature_tag: str = "general") -> str:
    start_time = time.time()
    try:
        client = get_groq_client()
        groq_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            groq_messages.append({"role": role, "content": msg["content"]})
        response = client.chat.completions.create(
            model=GROQ_MODEL, messages=groq_messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        output = response.choices[0].message.content

        elapsed = round(time.time() - start_time, 3)
        if elapsed >= ALERT_THRESHOLDS["latency_warning_sec"]:
            _alert_slow_response(feature_tag, elapsed)

        if _ls_enabled() and _ls_client:
            try:
                usage = response.usage
                run = get_current_run_tree()
                if run:
                    run.metadata.update({
                        "feature":        feature_tag,
                        "temperature":    temperature,
                        "max_tokens":     max_tokens,
                        "latency_sec":    elapsed,
                        "prompt_tokens":  getattr(usage, "prompt_tokens",     0),
                        "output_tokens":  getattr(usage, "completion_tokens", 0),
                        "total_tokens":   getattr(usage, "total_tokens",      0),
                    })
            except Exception:
                pass

        return output
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "429" in err:
            _alert_rate_limit(feature_tag)
            return "⚠️ Rate limit hit. Please wait a moment and retry."
        _alert_error(feature_tag, type(e).__name__, str(e))
        _increment_error_count(feature_tag)
        return f"⚠️ Error ({type(e).__name__}): {str(e)[:200]}"


# =========================================================
# ADZUNA JOBS API — reads credentials from .env
# =========================================================
ADZUNA_COUNTRY = "in"
ADZUNA_BASE    = "https://api.adzuna.com/v1/api/jobs"


def _get_adzuna_credentials():
    app_id  = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        st.error("ADZUNA_APP_ID or ADZUNA_APP_KEY not found. Please add them to your .env file.")
        st.stop()
    return app_id, app_key


def _simplify_query(role: str, exp_level: str) -> str:
    clean = re.sub(
        r"(?i)\s*(fresher|entry.?level|trainee|entry|intern(ship)?|graduate)\s*$", "", role
    ).strip()
    if not clean:
        clean = role
    modifier = {"fresher": "fresher", "junior": "junior", "mid": "", "senior": "senior"}.get(exp_level, "")
    combined = clean
    if modifier and modifier.lower() not in clean.lower():
        combined = f"{clean} {modifier}"
    return combined.strip()


def _adzuna_request(what: str, page: int, fetch_count: int, location: str) -> list:
    app_id, app_key = _get_adzuna_credentials()
    params = {
        "app_id": app_id, "app_key": app_key,
        "results_per_page": fetch_count, "what": what, "sort_by": "date",
    }
    if location and location.lower() not in ("india", "in", ""):
        params["where"] = location
    url  = f"{ADZUNA_BASE}/{ADZUNA_COUNTRY}/search/{page}"
    resp = requests.get(url, params=params, timeout=12)
    resp.raise_for_status()
    return resp.json().get("results", [])


def _adzuna_count(what: str) -> int:
    try:
        app_id, app_key = _get_adzuna_credentials()
        params = {
            "app_id": app_id, "app_key": app_key,
            "results_per_page": 1, "what": what,
        }
        url  = f"{ADZUNA_BASE}/{ADZUNA_COUNTRY}/search/1"
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json().get("count", 0)
    except Exception:
        return 0


def fetch_adzuna_jobs(role, location="india", results=7, page=1, exp_level="", exclude_urls=None):
    exclude_urls = exclude_urls or []
    max_results  = min(results, 7)
    fetch_count  = max_results + len(exclude_urls) + 3

    def _parse_jobs(raw_list):
        jobs = []
        for job in raw_list:
            redirect = job.get("redirect_url", "#")
            if redirect in exclude_urls:
                continue
            sal_min = job.get("salary_min")
            sal_max = job.get("salary_max")
            if sal_min and sal_max:
                sal_str = f"₹{int(sal_min):,} – ₹{int(sal_max):,}"
            elif sal_min:
                sal_str = f"₹{int(sal_min):,}+"
            else:
                sal_str = "Not disclosed"
            jobs.append({
                "title":       job.get("title", role),
                "company":     job.get("company", {}).get("display_name", "Unknown"),
                "location":    job.get("location", {}).get("display_name", "India"),
                "salary":      sal_str,
                "salary_min":  sal_min,
                "salary_max":  sal_max,
                "description": (job.get("description", "") or "")[:300].strip() + "…",
                "url":         redirect,
                "created":     (job.get("created", "") or "")[:10],
                "exp_level":   exp_level,
            })
            if len(jobs) >= max_results:
                break
        return jobs

    try:
        q1   = _simplify_query(role, exp_level)
        raw1 = _adzuna_request(q1, page, fetch_count, location)
        jobs = _parse_jobs(raw1)
        if len(jobs) >= 1:
            return jobs

        core = re.sub(r"(?i)\s*(fresher|junior|senior|intern(ship)?|trainee|associate|graduate|entry.?level)\s*", " ", role).strip()
        if core and core.lower() != q1.lower():
            raw2 = _adzuna_request(core, page, fetch_count, location)
            jobs = _parse_jobs(raw2)
            if len(jobs) >= 1:
                return jobs

        fallback_kw = core.split()[0] if core else role.split()[0]
        if fallback_kw.lower() not in (q1.lower(), core.lower()):
            raw3 = _adzuna_request(fallback_kw, 1, fetch_count, location)
            jobs = _parse_jobs(raw3)
            if jobs:
                return jobs
        return []

    except requests.exceptions.Timeout:
        _alert_adzuna_down()
        return [{"error": "⏱ Request timed out. Please try again."}]
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        if code in (500, 502, 503):
            _alert_adzuna_down()
        return [{"error": f"Adzuna API error {code}. Please try again."}]
    except Exception as e:
        return [{"error": f"Could not fetch jobs ({type(e).__name__}). Check your connection."}]


def _compute_job_match(job: dict, resume_skills: str) -> int:
    if not resume_skills:
        return 0
    jd_combined = (job.get("title", "") + " " + job.get("description", "")).lower()
    resume_tokens = set(re.sub(r"[^a-z0-9\s]", " ", resume_skills.lower()).split())
    jd_tokens = set(re.sub(r"[^a-z0-9\s]", " ", jd_combined).split())
    stop = {"and","or","the","to","of","in","for","with","a","an","is","are","be","at","by","on","as","we","you","our","your"}
    resume_tokens -= stop
    jd_tokens -= stop
    if not jd_tokens:
        return 0
    overlap = len(resume_tokens & jd_tokens)
    return min(int(overlap / max(len(jd_tokens), 1) * 300), 99)


def render_job_cards(jobs: list, role: str, location: str, show_applied_btn: bool = True,
                     resume_skills: str = "") -> None:
    if not jobs:
        st.warning(f"No jobs found for **'{role}'**. Try a shorter search term.")
        return
    if jobs and "error" in jobs[0]:
        st.warning(jobs[0]["error"])
        return

    applied_urls = [a["url"] for a in st.session_state.get("applied_jobs", [])]

    for i, job in enumerate(jobs, 1):
        already_applied = job["url"] in applied_urls
        card_class = "job-card-applied" if already_applied else "job-card"
        exp_tag = ""
        if job.get("exp_level"):
            label = {"fresher": "Fresher", "junior": "Junior", "mid": "Mid-Level", "senior": "Senior"}.get(job["exp_level"], "")
            if label:
                exp_tag = f'<span class="job-exp-tag">🎯 {label}</span><br>'

        applied_badge = '<span class="applied-badge">✓ Applied</span>' if already_applied else ""

        match_pct = _compute_job_match(job, resume_skills) if resume_skills else None
        match_html = ""
        if match_pct is not None:
            cls = "match-high" if match_pct >= 60 else ("match-mid" if match_pct >= 35 else "match-low")
            match_html = f'<span class="match-pill {cls}">⚡ {match_pct}% match</span>'

        st.markdown(f"""
        <div class="{card_class}">
            <div class="job-title">{i}. {job['title']} {applied_badge}{match_html}</div>
            {exp_tag}
            <div class="job-meta">🏢 {job['company']} &nbsp;·&nbsp; 📍 {job['location']}
            {'&nbsp;·&nbsp; 📅 ' + job['created'] if job.get('created') else ''}</div>
            <div class="job-salary">💰 {job['salary']}</div>
            <div class="job-desc">{job['description']}</div>
            <a class="job-apply" href="{job['url']}" target="_blank">Apply Now →</a>
        </div>
        """, unsafe_allow_html=True)

        if show_applied_btn and not already_applied:
            if st.button(f"✓ Mark as Applied — Job {i}", key=f"applied_btn_{i}_{job['url'][-12:]}", use_container_width=False):
                entry = {
                    "url": job["url"], "title": job["title"],
                    "company": job["company"], "date": job.get("created", ""),
                    "match_pct": match_pct or 0, "role_searched": role,
                }
                if job["url"] not in [a["url"] for a in st.session_state.applied_jobs]:
                    st.session_state.applied_jobs.append(entry)
                    st.session_state.applied_job_urls.append(job["url"])
                st.session_state.job_current_page += 1
                st.session_state.job_results = []
                st.success(f"Great! '{job['title']}' marked as applied. Loading fresh jobs…")
                st.rerun()


# =========================================================
# LLM WRAPPER
# =========================================================
# =========================================================
# LLM WRAPPER — all LLM calls now go through Groq (Llama-3.3-70B)
# HuggingFace inference removed — too slow / times out on Streamlit Cloud
# =========================================================

@traceable(name="groq_llm_call_safe", run_type="llm",
           metadata={"platform": "Smart CareerFordge", "model": "llama-3.3-70b-versatile"})
def safe_llm_invoke(prompt: str, fallback: str = "Model timed out — please retry.",
                    feature_tag: str = "general") -> str:
    """
    Drop-in replacement for the old HuggingFace safe_llm_invoke.
    Routes every call through Groq Llama-3.3-70B.
    10-20x faster than HuggingFace free inference; no cold-start timeouts.
    """
    start_time = time.time()
    try:
        result = _call_groq_direct(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a helpful career assistant. Be concise, accurate, and practical.",
            max_tokens=1024,
            temperature=0.3,
            feature_tag=feature_tag,
        )
        elapsed = round(time.time() - start_time, 3)

        if elapsed >= ALERT_THRESHOLDS["latency_warning_sec"]:
            _alert_slow_response(feature_tag, elapsed)

        if _ls_enabled():
            try:
                run = get_current_run_tree()
                if run:
                    run.metadata.update({
                        "feature":     feature_tag,
                        "latency_sec": elapsed,
                    })
            except Exception:
                pass

        return result
    except Exception as e:
        _alert_error(feature_tag, type(e).__name__, str(e))
        _increment_error_count(feature_tag)
        return f"{fallback}\n*(Error: {type(e).__name__})*"


# =========================================================
# DATABASE
# =========================================================
conn   = sqlite3.connect("career_path.db", check_same_thread=False)
cursor = conn.cursor()

def init_db():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS career_path (
            role TEXT PRIMARY KEY, skills TEXT, learning_roadmap TEXT,
            certifications TEXT, full_text TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, skills TEXT, full_text TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            feature   TEXT NOT NULL,
            rating    INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            comment   TEXT NOT NULL,
            tags      TEXT DEFAULT '',
            likes     INTEGER DEFAULT 0,
            created   TEXT DEFAULT (datetime('now','localtime'))
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_likes (
            feedback_id INTEGER NOT NULL,
            session_id  TEXT NOT NULL,
            PRIMARY KEY (feedback_id, session_id)
        )""")
    conn.commit()

init_db()


# =========================================================
# PDF PARSER
# =========================================================
@st.cache_resource
def load_and_parse_pdf(pdf_path="Career_Path.pdf"):
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    all_lines = []
    for pi, pt in enumerate(pages):
        for li, line in enumerate(pt.page_content.split("\n")):
            all_lines.append((pi, li, line))

    role_header_pattern = re.compile(r"^#{0,3}\s*\*{0,2}\s*role\s*:\*{0,2}\s*(.*)", re.IGNORECASE)
    false_positive = re.compile(r"(?i)(based access|transition|http|youtube|github)")

    role_entries = []
    for flat_idx, (pi, li, line) in enumerate(all_lines):
        m = role_header_pattern.match(line.strip())
        if not m:
            continue
        candidate = m.group(1).strip().strip("*").strip()
        if not candidate or re.fullmatch(r"[\s\-\*#:]*", candidate):
            for nf in range(flat_idx + 1, min(flat_idx + 5, len(all_lines))):
                nl = all_lines[nf][2].strip().strip("*").strip()
                if nl and not re.match(r"(?i)(skills|learning|certif|#|-{3})", nl):
                    candidate = nl
                    break
        candidate = re.sub(r"[\*#]+", "", candidate).strip()
        if len(candidate) < 3 or false_positive.search(candidate):
            continue
        role_entries.append((flat_idx, candidate))

    def extract_field(block, *patterns):
        section_boundary = re.compile(
            r"(?i)(?:#{0,3}\s*\*{0,2}\s*)(?:skills|learning roadmap|learning path|certificates?|related roadmaps?)[\s:\*]*")
        for pat in patterns:
            m = re.search(pat, block, re.DOTALL | re.IGNORECASE)
            if m:
                start    = m.end()
                next_sec = section_boundary.search(block, start)
                end      = next_sec.start() if next_sec else len(block)
                raw      = block[start:end].strip()
                raw      = re.sub(r"[\*#]{2,}", "", raw)
                raw      = re.sub(r"^-{3,}$", "", raw, flags=re.MULTILINE)
                raw      = re.sub(r"\n{3,}", "\n\n", raw)
                return raw.strip()
        return "Not listed"

    full_text_joined = "\n".join(l for (_, _, l) in all_lines)
    line_char_offsets, pos = [], 0
    for _, _, l in all_lines:
        line_char_offsets.append(pos)
        pos += len(l) + 1

    role_data_map, role_names_list = {}, []
    for entry_idx, (flat_idx, role_name) in enumerate(role_entries):
        block_start = line_char_offsets[flat_idx]
        block_end   = (line_char_offsets[role_entries[entry_idx + 1][0]]
                       if entry_idx + 1 < len(role_entries) else len(full_text_joined))
        block = full_text_joined[block_start:block_end]
        skills  = extract_field(block, r"(?:#{0,3}\s*\*{0,2}\s*)skills?\s*[\*:\s]+")
        skills  = " ".join(skills.split("\n")).strip()
        skills  = re.sub(re.escape(role_name), "", skills, flags=re.IGNORECASE).strip(" ,")
        roadmap = extract_field(block, r"(?:#{0,3}\s*\*{0,2}\s*)(?:learning roadmap|learning path)\s*[\*:\s]+")
        certs   = extract_field(block, r"(?:#{0,3}\s*\*{0,2}\s*)certificates?\s*[\*:\s]+")
        if role_name not in role_data_map:
            role_data_map[role_name] = {
                "skills": skills or "Not listed", "learning_roadmap": roadmap or "Not listed",
                "certifications": certs or "Not listed", "full_text": block[:3000]
            }
            role_names_list.append(role_name)

    return role_names_list, role_data_map


role_names, role_data_map = load_and_parse_pdf()
for rname, rdata in role_data_map.items():
    cursor.execute("INSERT OR REPLACE INTO career_path VALUES (?,?,?,?,?)",
                   (rname, rdata["skills"], rdata["learning_roadmap"],
                    rdata["certifications"], rdata["full_text"]))
conn.commit()


# =========================================================
# EMBEDDINGS + VECTOR DB
# =========================================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  model_kwargs={"device": "cpu"})

embeddings = get_embeddings()

@st.cache_resource
def build_role_vector_db(_role_names, _role_data_map):
    texts = [_role_data_map[r]["full_text"] for r in _role_names]
    ids   = [str(uuid.uuid4()) for _ in _role_names]
    metas = [{"role": r} for r in _role_names]
    return Chroma.from_texts(texts=texts, embedding=embeddings, ids=ids, metadatas=metas,
                              collection_name="all_roles", persist_directory="./chroma_roles")

role_vectordb = build_role_vector_db(role_names, role_data_map)


# =========================================================
# DYNAMIC SKILL DICTIONARIES
# =========================================================
def _raw_clean(s: str) -> str:
    s = s.lower().strip().strip("*-•/()")
    s = re.sub(r"[^a-z0-9\s/\.\+#]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _collect_all_pdf_skills(rdmap: dict) -> list:
    seen, out = set(), []
    for rdata in rdmap.values():
        raw = rdata.get("skills", "")
        if not raw or raw == "Not listed":
            continue
        for tok in re.split(r"[,\n;|•]", raw):
            t = _raw_clean(tok)
            if len(t) > 1 and t not in seen:
                seen.add(t)
                out.append(t)
    return out

@st.cache_resource
def build_dynamic_skill_dicts(_role_data_map: dict):
    SKILL_ALIASES = {
        "js": "javascript", "javascript": "js", "ts": "typescript", "typescript": "ts",
        "py": "python", "k8s": "kubernetes", "kubernetes": "k8s", "tf": "terraform",
        "terraform": "tf", "ci/cd": "continuous integration",
        "continuous integration": "ci/cd", "ml": "machine learning",
        "machine learning": "ml", "dl": "deep learning", "deep learning": "dl",
        "ai": "artificial intelligence", "artificial intelligence": "ai",
        "nlp": "natural language processing", "natural language processing": "nlp",
        "cv": "computer vision", "computer vision": "cv",
        "sql": "structured query language", "nosql": "non relational database",
        "api": "rest api", "rest api": "api", "devops": "development operations",
        "qa": "quality assurance", "ui/ux": "user interface design",
        "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
        "llm": "large language models", "large language models": "llm",
        "rag": "retrieval augmented generation",
        "retrieval augmented generation": "rag", "node.js": "nodejs",
        "nodejs": "node.js", "react.js": "react", "vue.js": "vue",
        "next.js": "nextjs", "spring boot": "spring",
    }
    IMPLIES = {
        "pytorch": ["deep learning", "neural networks", "machine learning", "python"],
        "tensorflow": ["deep learning", "neural networks", "machine learning", "python"],
        "scikit-learn": ["machine learning", "data preprocessing", "python"],
        "langchain": ["llm", "prompt engineering", "rag", "api", "python"],
        "docker": ["containerization", "devops", "microservices", "linux"],
        "kubernetes": ["containerization", "devops", "orchestration", "docker", "linux"],
        "react": ["javascript", "frontend", "html", "css"],
        "nodejs": ["javascript", "backend", "rest api"],
        "django": ["python", "backend", "rest api"],
        "pandas": ["data analysis", "python", "data wrangling"],
        "aws": ["cloud", "devops"], "azure": ["cloud", "devops"], "gcp": ["cloud", "devops"],
        "python": ["programming", "scripting"], "java": ["programming", "oop"],
        "javascript": ["programming", "web development"],
    }
    pdf_skills = _collect_all_pdf_skills(_role_data_map)
    STRONG_TECH_KW = set(pdf_skills)
    for k in SKILL_ALIASES:
        if len(k) > 1: STRONG_TECH_KW.add(k)
    for k in IMPLIES: STRONG_TECH_KW.add(k)
    return SKILL_ALIASES, IMPLIES, STRONG_TECH_KW

SKILL_ALIASES, IMPLIES, STRONG_TECH_KEYWORDS = build_dynamic_skill_dicts(role_data_map)


# =========================================================
# SKILL MATCHING HELPERS
# =========================================================
def clean_skill(s):
    s = s.lower().strip().strip("*-•()")
    s = re.sub(r"[^a-z0-9\s/\.\+#]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_skill_list(raw_text):
    parts = re.split(r"[,\n;|•]", raw_text)
    skills, seen = [], set()
    for p in parts:
        p = clean_skill(p)
        if len(p) > 1 and p not in seen:
            seen.add(p); skills.append(p)
    return skills

def parse_resume_skill_list(raw_text):
    cleaned = re.sub(r"(?m)^[A-Za-z0-9 ,&/()\.\-]+:\s*", " ", raw_text)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    parts   = re.split(r"[,\n;|•]", cleaned)
    skills, seen = [], set()
    for p in parts:
        p = clean_skill(p)
        if len(p) > 1 and p not in seen:
            seen.add(p); skills.append(p)
    return skills

def expand_resume_skills(skill_list):
    expanded = set(skill_list)
    queue, visited = list(skill_list), set()
    while queue:
        s = queue.pop()
        if s in visited: continue
        visited.add(s)
        alias = SKILL_ALIASES.get(s)
        if alias and alias not in expanded:
            expanded.add(alias); queue.append(alias)
        for implied in IMPLIES.get(s, []):
            imp = clean_skill(implied)
            if imp not in expanded:
                expanded.add(imp); queue.append(imp)
                a2 = SKILL_ALIASES.get(imp)
                if a2: expanded.add(a2)
    return expanded

def token_overlap_score(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def skill_is_matched(role_skill, resume_expanded_set, resume_skill_list):
    rs = clean_skill(role_skill)
    if rs in resume_expanded_set: return True
    alias = SKILL_ALIASES.get(rs, rs)
    if alias in resume_expanded_set: return True
    for res_sk in resume_skill_list:
        if token_overlap_score(rs, res_sk) >= 0.35: return True
        if token_overlap_score(alias, res_sk) >= 0.35: return True
    for res_sk in resume_expanded_set:
        if rs in res_sk or res_sk in rs: return True
    return False

def match_skills(role_skills_raw, resume_skills_raw, emb_model):
    role_list   = parse_skill_list(role_skills_raw)
    resume_list = parse_resume_skill_list(resume_skills_raw)
    resume_exp  = expand_resume_skills(resume_list)
    matched, needs_embed = [], []
    for rs in role_list:
        if skill_is_matched(rs, resume_exp, resume_list):
            matched.append(rs)
        else:
            needs_embed.append(rs)
    if needs_embed and resume_list:
        role_vecs   = emb_model.embed_documents(needs_embed)
        resume_vecs = emb_model.embed_documents(resume_list)
        for i, rv in enumerate(role_vecs):
            sims = cosine_similarity([rv], resume_vecs)[0]
            if sims.max() >= 0.52:
                matched.append(needs_embed[i])
    missing = [r for r in role_list if r not in matched]
    score   = int(len(matched) / max(len(role_list), 1) * 100)
    return matched, missing, score

def normalize_skills(text):
    if not text: return set()
    return set(parse_resume_skill_list(text))


# =========================================================
# HELPERS & BUSINESS LOGIC
# =========================================================
def get_role_data(role_name):
    if role_name in role_data_map: return role_data_map[role_name]
    cursor.execute("SELECT * FROM career_path WHERE role=?", (role_name,))
    row = cursor.fetchone()
    if row:
        return {"skills": row[1], "learning_roadmap": row[2],
                "certifications": row[3], "full_text": row[4]}
    return None

def extract_resume_skills(resume_text):
    for label in ["Technical Skills","Skills Summary","Core Skills","Key Skills",
                  "Skills & Technologies","Skills","Tools & Technologies","Technologies","Competencies"]:
        pat = rf"(?i){label}\s*[:\n](.*?)(?=\n[A-Z][A-Za-z& ]{{2,}}\s*\n|\Z)"
        m   = re.search(pat, resume_text, re.DOTALL)
        if m:
            raw = m.group(1).strip()
            if len(raw) > 20: return raw
    return resume_text[:1000].strip()

def process_resume(pdf_path):
    loader    = PyPDFLoader(pdf_path)
    pages     = loader.load()
    full_text = "\n".join([p.page_content for p in pages])
    skills    = extract_resume_skills(full_text)
    return full_text, skills

def llm_ats_analysis(resume_full_text, jd_text):
    resume_skills_raw = extract_resume_skills(resume_full_text)
    try:
        matched, missing, real_score = match_skills(
            jd_text[:2000], resume_skills_raw, embeddings
        )
        matched_str = ", ".join(matched[:12]) if matched else "none detected"
        missing_str = ", ".join(missing[:12]) if missing else "none detected"
        score_note  = (f"COMPUTED skill-overlap score: {real_score}% "
                       f"(from deterministic token + semantic matching — do not change this number)")
    except Exception:
        real_score  = None
        matched_str = "could not compute"
        missing_str = "could not compute"
        score_note  = "Skill overlap: could not be pre-computed — estimate carefully."

    score_instruction = (
        f"Use {real_score} as the Overall ATS Score (±5 only for experience/education fit)."
        if real_score is not None
        else "Estimate a score 0-100 based only on the text below."
    )

    prompt = f"""You are an ATS expert. Analyze the resume against the job description.

PRE-COMPUTED DATA — treat these as facts, do not alter them:
- {score_note}
- Matched skills: {matched_str}
- Missing skills: {missing_str}

{score_instruction}

Structure your response EXACTLY like this (no extra sections):

Overall ATS Score: <number>
Decision: <Excellent Fit / Good Fit / Weak Fit / Not a Fit>

Skills Score: <number>/100
Experience Score: <number>/100
Education Score: <number>/100

Missing Skills:
{missing_str}

Resume Improvement Suggestions:
- <suggestion referencing actual missing skills above>
- <suggestion 2>
- <suggestion 3>

Reasoning:
<2-3 sentences that reference the actual matched/missing skills listed above>

STRICT RULES:
- Do NOT invent skills not listed in the resume or JD
- Do NOT change the pre-computed matched/missing skill lists
- Do NOT add URLs or course links

Resume:
{resume_full_text[:2000]}

Job Description:
{jd_text[:1500]}
"""
    return safe_llm_invoke(prompt, feature_tag="ats_analysis")


# =========================================================
# RESUME SCORE CARD
# =========================================================
def compute_resume_score_card(resume_text: str, resume_skills: str) -> dict:
    text = resume_text.lower()

    skill_count  = len(normalize_skills(resume_skills))
    skills_score = min(skill_count * 2, 25)

    kw_hits    = sum(1 for kw in STRONG_TECH_KEYWORDS
                     if re.search(r"\b" + re.escape(kw) + r"\b", text))
    tech_score = min(kw_hits, 20)

    impact_words = ["built", "developed", "designed", "implemented", "led", "improved",
                    "delivered", "automated", "optimized", "architected", "reduced", "increased",
                    "managed", "created", "deployed", "integrated", "launched"]
    project_score = min(sum(1 for w in impact_words if w in text) * 2, 20)

    exp_score = 0
    if re.search(r"\b(\d+)\+?\s*(years?|months?)\b", text): exp_score = 12
    if "internship" in text or "intern " in text: exp_score = max(exp_score, 8)
    if "fresher" in text or "graduate" in text: exp_score = max(exp_score, 5)
    exp_score = min(exp_score, 15)

    cert_words = ["certified","certification","certificate","coursera","udemy","aws certified",
                  "google certified","azure certified","nptel","coursework","training"]
    cert_score = min(sum(1 for w in cert_words if w in text) * 3, 10)

    section_words = ["education","experience","skills","projects","objective","summary",
                     "achievements","contact","email","linkedin"]
    struct_score = min(sum(1 for w in section_words if w in text) * 1.5, 10)

    total = int(skills_score + tech_score + project_score + exp_score + cert_score + struct_score)
    total = max(0, min(100, total))

    if total >= 85:   grade, label, css = "A", "Excellent Resume", "score-card-a"
    elif total >= 70: grade, label, css = "B", "Good Resume", "score-card-b"
    elif total >= 50: grade, label, css = "C", "Average Resume", "score-card-c"
    else:             grade, label, css = "D", "Needs Improvement", "score-card-d"

    return {
        "total": total, "grade": grade, "label": label, "css": css,
        "breakdown": {
            "Skills Breadth":    (int(skills_score),  25),
            "Tech Keywords":     (int(tech_score),     20),
            "Impact & Projects": (int(project_score),  20),
            "Experience":        (int(exp_score),       15),
            "Certifications":    (int(cert_score),      10),
            "Structure":         (int(struct_score),    10),
        }
    }


# =========================================================
# AI COVER LETTER GENERATOR (Groq)
# =========================================================
def generate_cover_letter(resume_text: str, jd_text: str, candidate_name: str,
                           exp_level: str, location: str) -> str:
    system = (
        "You are an expert cover letter writer. "
        "You ONLY use information explicitly present in the resume provided. "
        "You NEVER invent job titles, company names, metrics, or achievements not in the resume."
    )
    prompt = f"""Write a professional cover letter for this candidate.

Candidate Name: {candidate_name or 'the applicant'}
Experience Level: {exp_level}
Location: {location}

Resume (source of truth — only use what is here):
{resume_text[:1500]}

Job Description:
{jd_text[:1200]}

Instructions:
- 3-4 paragraphs, professional tone
- Opening: hook with enthusiasm for the specific role
- Body 1: match 2-3 key skills/experiences FROM THE RESUME ABOVE to JD requirements
- Body 2: highlight a specific achievement or project that ACTUALLY APPEARS in the resume
- Closing: call to action, availability, contact intent
- DO NOT use generic phrases like "I am writing to apply"
- DO NOT invent metrics (e.g. "increased sales by 40%") unless they appear in the resume
- DO NOT mention companies, tools, or roles not present in the resume text
- Keep it under 350 words
"""
    return _call_groq_direct(
        [{"role": "user", "content": prompt}], system,
        max_tokens=800, temperature=0.6, feature_tag="cover_letter"
    )


# =========================================================
# SALARY ESTIMATOR (Groq)
# =========================================================
def estimate_salary(skills: str, location: str, exp_years: int, exp_level: str,
                    target_role: str) -> str:
    real_salary_lines = []
    try:
        live_jobs = fetch_adzuna_jobs(target_role, "india", results=5)
        for j in live_jobs:
            if j.get("salary_min") and j.get("salary_max"):
                avg_lpa = round((j["salary_min"] + j["salary_max"]) / 2 / 100000, 1)
                real_salary_lines.append(
                    f"  • {j['company']} — ₹{avg_lpa} LPA (₹{int(j['salary_min']):,}–₹{int(j['salary_max']):,})"
                )
    except Exception:
        pass

    if real_salary_lines:
        grounding_block = (
            "LIVE MARKET DATA (from real job listings — use these as anchors for your ranges):\n"
            + "\n".join(real_salary_lines[:5])
        )
        data_note = "Base your range on the LIVE DATA above. State ranges consistent with it."
    else:
        grounding_block = "No live listings available at this time."
        data_note = (
            "No live data available. Provide estimates but prefix EVERY number with 'approx.' "
            "and add a disclaimer that figures are estimates only."
        )

    system = (
        "You are a compensation analyst. "
        "Use only the data provided to you. "
        "Never present invented numbers as fact. "
        "If data is unavailable for a city, say 'data unavailable' — do not guess."
    )
    prompt = f"""Estimate salary for this candidate in India.

{grounding_block}

Target Role: {target_role}
Experience Level: {exp_level} ({exp_years} years)
Location: {location}
Key Skills: {skills[:600]}

{data_note}

Provide:
1. **Expected CTC Range**: Low – High (in LPA) — anchored to live data above if available
2. **Median Market Rate**: Single figure
3. **Salary by City**: Bangalore / Hyderabad / Mumbai / Pune / Remote
4. **Negotiation Tips**: 2-3 bullet points specific to this skill set
5. **Growth Projection**: Expected CTC after 2-3 years

STRICT RULES:
- Do NOT invent salary numbers without basis
- Do NOT mention specific companies not in the live data above
- Qualify all figures with "approx." if no live data was available
- Use LPA (Lakhs Per Annum) as unit throughout
"""
    return _call_groq_direct(
        [{"role": "user", "content": prompt}], system,
        max_tokens=700, temperature=0.1, feature_tag="salary_estimator"
    )


# =========================================================
# LINKEDIN SUMMARY WRITER (Groq)
# =========================================================
def generate_linkedin_bio(resume_text: str, resume_skills: str, exp_level: str,
                           target_role: str, candidate_name: str) -> str:
    system = (
        "You are a LinkedIn profile specialist. "
        "You write only from what the candidate's resume contains. "
        "You NEVER invent companies, job titles, metrics, projects, or achievements "
        "that are not explicitly stated in the resume excerpt provided."
    )
    prompt = f"""Write a LinkedIn 'About' section for this candidate.

Candidate: {candidate_name or 'the candidate'}
Target Role: {target_role}
Experience Level: {exp_level}
Skills (from resume): {resume_skills[:500]}
Resume Excerpt (source of truth): {resume_text[:800]}

Requirements:
- First person, conversational yet professional
- Start with a strong hook (NOT "I am a...")
- 3-4 short paragraphs: current focus → key skills → notable achievements → what you're seeking
- Include 5-8 relevant skill keywords naturally (for LinkedIn SEO)
- End with a call to action
- 150-220 words total
- Tone: confident, human, not robotic

GROUNDING RULES (strictly follow):
- Only reference skills that appear in the Skills or Resume Excerpt above
- If the resume has limited achievements, write aspirationally about what the candidate is building toward
- Do NOT invent metrics like "increased performance by 30%" unless in the resume
- Do NOT mention employer names not in the resume excerpt
"""
    return _call_groq_direct(
        [{"role": "user", "content": prompt}], system,
        max_tokens=600, temperature=0.6, feature_tag="linkedin_bio"
    )


# =========================================================
# JOB MARKET INSIGHTS (Adzuna counts)
# =========================================================
def fetch_market_insights(roles_to_check: list) -> dict:
    insights = {}
    for role in roles_to_check[:8]:
        count = _adzuna_count(role)
        insights[role] = count
    return insights


# =========================================================
# SALARY TRENDS CHART
# =========================================================
def compute_salary_trends(top_roles_data: list, resume_skills: str) -> dict:
    SALARY_MAP = {
        "Data Scientist": 12, "Machine Learning Engineer": 14, "AI Engineer": 15,
        "Data Analyst": 8, "Data Engineer": 13, "Business Intelligence Analyst": 9,
        "Software Engineer": 10, "Backend Developer": 11, "Frontend Developer": 9,
        "Full Stack Developer": 12, "DevOps Engineer": 13, "Cloud Engineer": 14,
        "Site Reliability Engineer": 15, "Cybersecurity Analyst": 10,
        "Mobile Developer": 10, "Python Developer": 11, "Java Developer": 10,
        "React Developer": 10, "MLOps Engineer": 14, "NLP Engineer": 13,
        "Computer Vision Engineer": 13, "Blockchain Developer": 14,
        "Product Manager": 16, "Scrum Master": 11, "QA Engineer": 8,
    }
    result = {}
    for role_tuple in top_roles_data[:5]:
        role_name = role_tuple[0]
        match_pct = role_tuple[1]

        try:
            jobs_sample = fetch_adzuna_jobs(role_name, "india", results=5)
            salaries = []
            for j in jobs_sample:
                if j.get("salary_min") and j.get("salary_max"):
                    avg_inr = (j["salary_min"] + j["salary_max"]) / 2
                    avg_lpa = round(avg_inr / 100000, 1)
                    salaries.append(avg_lpa)
            if salaries:
                result[role_name] = round(float(np.mean(salaries)), 1)
                continue
        except Exception:
            pass

        base_lpa = SALARY_MAP.get(role_name, 10)
        adjusted = base_lpa * (0.8 + 0.4 * match_pct / 100)
        result[role_name] = round(adjusted, 1)

    return result


# =========================================================
# EXISTING HELPERS
# =========================================================
def compute_section1_scores(skills_text):
    cleaned   = re.sub(r"(?m)^[A-Za-z0-9 ,&/()\.\-]+:\s*", " ", skills_text)
    cleaned   = re.sub(r"\([^)]*\)", " ", cleaned)
    raw_items = [clean_skill(p) for p in re.split(r"[,\n;|•]", cleaned)
                 if clean_skill(p) and len(clean_skill(p)) > 1]
    unique_items = list(dict.fromkeys(raw_items))
    density      = round(len(unique_items) / max(len(raw_items), 1) * 100, 1)
    text_lower   = skills_text.lower()
    kw_hits = sum(1 for kw in STRONG_TECH_KEYWORDS
                  if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
    return density, kw_hits

def compute_section2_scores(role_name, rdata, emb_model):
    fields  = [rdata.get("skills",""), rdata.get("learning_roadmap",""), rdata.get("certifications","")]
    filled  = sum(1 for f in fields if f and f != "Not listed")
    return round(filled / 3 * 100)

def compute_section3_scores(role_skills_raw, resume_skills_raw, matched, missing, emb_model):
    role_list   = parse_skill_list(role_skills_raw)
    resume_list = parse_resume_skill_list(resume_skills_raw)
    resume_exp  = expand_resume_skills(resume_list)
    role_set    = {clean_skill(s) for s in role_list}
    jaccard     = round(len(resume_exp & role_set) / max(len(resume_exp | role_set), 1), 3)
    sem_score   = None
    if role_list and resume_list:
        try:
            role_vecs   = emb_model.embed_documents(role_list)
            resume_vecs = emb_model.embed_documents(resume_list)
            best_matches = [cosine_similarity([v], resume_vecs)[0].max() for v in role_vecs]
            sem_score = round(float(np.mean(best_matches)), 3)
        except Exception: pass
    n_matched = len(matched)
    precision = round(n_matched / max(len(resume_list), 1) * 100, 1)
    recall    = round(n_matched / max(len(role_list),   1) * 100, 1)
    return jaccard, sem_score, precision, recall

def compute_section4_scores(top_roles_data, resume_skills_raw, emb_model):
    resume_list = parse_resume_skill_list(resume_skills_raw)
    resume_clean_joined = ", ".join(resume_list)
    resume_set  = set(resume_list)
    top_jaccard = None
    if top_roles_data:
        top_rd = get_role_data(top_roles_data[0][0])
        if top_rd and top_rd["skills"] != "Not listed":
            top_set = set(parse_skill_list(top_rd["skills"]))
            top_jaccard = round(len(resume_set & top_set) / max(len(resume_set | top_set), 1), 3)
    median_cosine = None
    skill_texts = [get_role_data(r)["skills"] for r, *_ in top_roles_data
                   if get_role_data(r) and get_role_data(r)["skills"] != "Not listed"]
    if skill_texts:
        try:
            res_vec   = emb_model.embed_documents([resume_clean_joined])[0]
            role_vecs = emb_model.embed_documents(skill_texts)
            cosines   = [float(cosine_similarity([res_vec], [rv])[0][0]) for rv in role_vecs]
            median_cosine = round(float(np.median(cosines)), 3)
        except Exception: pass
    return top_jaccard, median_cosine

def compute_section5_scores(resume_full_text, jd_text, resume_skills_raw, emb_model):
    sem_match = None
    try:
        rv  = emb_model.embed_documents([resume_full_text[:3000]])[0]
        jdv = emb_model.embed_documents([jd_text[:3000]])[0]
        sem_match = round(float(cosine_similarity([rv], [jdv])[0][0]), 3)
    except Exception: pass
    jd_coverage = None
    jd_cleaned = re.sub(r"(?i)(responsibilities|requirements|qualifications|job description|"
                        r"we are looking|you will|your role|about us|preferred|nice to have|"
                        r"must have|minimum)[^\n]*", "", jd_text)
    jd_skill_list = parse_skill_list(jd_cleaned)
    if len(jd_skill_list) >= 3:
        try:
            _, _, jd_coverage = match_skills(", ".join(jd_skill_list), resume_skills_raw, emb_model)
        except Exception: pass
    return sem_match, jd_coverage

def compute_section6_scores(transitions, current_role, resume_skills_raw):
    if not transitions: return None, None, None
    gap          = round(100 - float(np.mean([t["match_percent"] for t in transitions])), 1)
    missing_load = round(float(np.mean([len(t["missing_skills"]) for t in transitions])), 1)
    resume_exp  = expand_resume_skills(parse_resume_skill_list(resume_skills_raw))
    cur_rdata   = get_role_data(current_role)
    cur_exp     = (expand_resume_skills(parse_skill_list(cur_rdata["skills"]))
                   if cur_rdata and cur_rdata["skills"] != "Not listed" else set())
    blended     = resume_exp | cur_exp
    realistic_overlap = None
    top_rdata = get_role_data(transitions[0]["target_role"])
    if top_rdata and top_rdata["skills"] != "Not listed":
        target_exp = expand_resume_skills(parse_skill_list(top_rdata["skills"]))
        realistic_overlap = round(len(blended & target_exp) / max(len(blended | target_exp), 1) * 100, 1)
    return gap, missing_load, realistic_overlap

def resume_based_transitions(resume_skills_text, current_role, top_k=3):
    if not resume_skills_text: return []
    results = []
    for role in role_names:
        if role.lower() == current_role.lower(): continue
        rdata = get_role_data(role)
        if not rdata or not rdata["skills"] or rdata["skills"] == "Not listed": continue
        matched, missing, match_pct = match_skills(rdata["skills"], resume_skills_text, embeddings)
        results.append({"target_role": role, "match_percent": match_pct,
                        "matched_skills": matched, "missing_skills": missing})
    results.sort(key=lambda x: x["match_percent"], reverse=True)
    return results[:top_k]

def llm_transition_reason(resume_text, current_role, target_role, match_pct=None):
    if match_pct is not None:
        if match_pct < 35:
            honesty = (
                f"The skill match is LOW ({match_pct}%). "
                "Be HONEST — your first bullet must clearly acknowledge the significant skill gap. "
                "Do not make this transition sound easier than it is."
            )
        elif match_pct < 60:
            honesty = (
                f"The skill match is MODERATE ({match_pct}%). "
                "Be BALANCED — mention both existing strengths and what still needs work."
            )
        else:
            honesty = (
                f"The skill match is STRONG ({match_pct}%). "
                "Highlight the genuine alignment between the candidate's background and target role."
            )
    else:
        honesty = "Be honest and balanced about the fit."

    prompt = (
        f"Resume:\n{resume_text[:1500]}\n\n"
        f"Current Role: {current_role}\n"
        f"Target Role: {target_role}\n\n"
        f"{honesty}\n\n"
        f"Give 4-5 bullet points explaining this career transition.\n\n"
        f"STRICT RULES:\n"
        f"- Only reference skills and experience that ACTUALLY APPEAR in the resume above\n"
        f"- Do NOT invent tools, projects, or achievements not mentioned\n"
        f"- Do NOT add URLs, course names, or certification links\n"
        f"- If fit is weak, say so clearly in at least one bullet"
    )
    return safe_llm_invoke(prompt, feature_tag="transition_reason")

def llm_learning_roadmap(target_role, missing_skills):
    rdata = get_role_data(target_role)
    real_roadmap_block = ""
    if rdata and rdata.get("learning_roadmap") and rdata["learning_roadmap"] != "Not listed":
        real_roadmap_block = (
            f"\nVERIFIED ROADMAP FROM KNOWLEDGE BASE "
            f"(use this as the foundation for your structure):\n"
            f"{rdata['learning_roadmap'][:800]}\n"
        )

    prompt = (
        f"Target Role: {target_role}\n"
        f"Skills to Learn: {', '.join(missing_skills[:15])}\n"
        f"{real_roadmap_block}\n"
        f"Create a Beginner → Intermediate → Advanced structured roadmap.\n\n"
        f"STRICT RULES:\n"
        f"- Do NOT invent or fabricate any URLs or hyperlinks\n"
        f"- Do NOT make up specific course titles (e.g. 'Python Bootcamp by X')\n"
        f"- Instead say 'search for X on Coursera/Udemy/YouTube' — never give a fake link\n"
        f"- Focus on WHAT to learn and in WHAT ORDER\n"
        f"- Include 2-3 realistic project ideas relevant to {target_role}\n"
        f"- Timeline estimates should use ranges: 'typically 4-8 weeks', not exact days"
    )
    return safe_llm_invoke(prompt, feature_tag="learning_roadmap")

def suggest_top_3_roles(resume_skills_text, top_k=3):
    resume_clean = ", ".join(parse_resume_skill_list(resume_skills_text))
    role_scores  = []
    for role in role_names:
        rdata = get_role_data(role)
        if not rdata or rdata["skills"] == "Not listed": continue
        matched, missing, cascade_pct = match_skills(rdata["skills"], resume_skills_text, embeddings)
        try:
            rv = embeddings.embed_documents([resume_clean])[0]
            sv = embeddings.embed_documents([rdata["skills"]])[0]
            blob_score = float(cosine_similarity([rv], [sv])[0][0])
        except Exception:
            blob_score = 0.0
        combined = 0.7 * cascade_pct + 0.3 * blob_score * 100
        role_scores.append((role, cascade_pct, combined, matched, missing))
    role_scores.sort(key=lambda x: x[2], reverse=True)
    return role_scores[:top_k]

def compute_resume_readiness(resume_text, resume_skills_text):
    text = resume_text.lower()
    skill_score      = min(len(normalize_skills(resume_skills_text)) * 8, 100)
    project_score    = min(sum(k in text for k in ["project","capstone","built","developed","mini project"]) * 20, 100)
    experience_score = 0
    if re.search(r"\b(\d+)\+?\s*(years?|months?)\b", text): experience_score = 70
    if "internship" in text or "experience" in text: experience_score = max(experience_score, 60)
    tools_score = min(sum(k in text for k in [
        "python","java","sql","aws","docker","tensorflow","pytorch","flask","react",
        "linux","git","javascript","typescript","kubernetes","spark","kafka","nodejs"]) * 5, 100)
    communication_score = min(sum(k in text for k in [
        "led","implemented","designed","collaborated","presented","managed",
        "improved","delivered","architected","optimized","reduced","increased","automated"]) * 10, 100)
    learning_score = min(sum(k in text for k in [
        "certification","certified","course","training","workshop",
        "coursera","udemy","nptel","aws certified"]) * 20, 100)
    return {
        "Skills Strength": skill_score, "Projects": project_score,
        "Experience": experience_score, "Tools & Tech": tools_score,
        "Communication": communication_score, "Learning Mindset": learning_score,
    }

def generate_career_report(candidate_name, resume_skills, ats_result_text, top_roles, transitions, current_role):
    report = [
        "CAREER INTELLIGENCE REPORT", "=" * 40,
        f"\nCandidate: {candidate_name}",
        "\n--- RESUME SKILLS ---", resume_skills,
        "\n--- ATS ANALYSIS ---", ats_result_text,
        "\n--- TOP ROLE RECOMMENDATIONS ---",
    ]
    for role, score in top_roles:
        report.append(f"{role} -> Match: {score * 100:.2f}%")
    report.append("\n--- CAREER TRANSITION INSIGHTS ---")
    for t in transitions:
        report.append(f"\nTransition: {current_role} -> {t['target_role']}")
        report.append(f"Skill Match: {t['match_percent']}%")
        for s in t["matched_skills"]: report.append(f"  + {s}")
        for s in t["missing_skills"]:  report.append(f"  - {s}")
    return "\n".join(report)

def _badge(label, value, good_thresh, warn_thresh, fmt="{}", reverse=False, caption=""):
    if value is None:
        st.metric(label, "N/A")
        if caption: st.caption(caption)
        return
    v = float(value)
    status = ("Good" if (v <= good_thresh if reverse else v >= good_thresh)
              else ("Fair" if (v <= warn_thresh if reverse else v >= warn_thresh) else "Low"))
    st.metric(f"[{status}] {label}", fmt.format(value))
    if caption: st.caption(caption)


# =========================================================
# FEEDBACK SYSTEM — DB HELPERS
# =========================================================
FEEDBACK_FEATURES = [
    "Resume Analyzer", "Roles Explorer", "Skill Match & Gap",
    "Top Role Picks", "Career Transitions", "ATS Analysis",
    "Readiness Dashboard", "Career Assistant", "Job Listings",
    "Job Market Insights", "Salary Trends", "AI Cover Letter",
    "Salary Estimator", "LinkedIn Bio Writer", "General / Overall",
]

FEEDBACK_TAGS = [
    "Very Accurate", "Saved Time", "Easy to Use", "Needs Improvement",
    "Great AI", "Helpful Roadmap", "Job Match Useful", "Loved the UI",
    "Salary Data Helpful", "Would Recommend",
]

def fb_submit(name: str, feature: str, rating: int, comment: str, tags: list) -> bool:
    try:
        tags_str = ",".join(tags)
        cursor.execute(
            "INSERT INTO feedback (name, feature, rating, comment, tags) VALUES (?,?,?,?,?)",
            (name.strip() or "Anonymous", feature, rating, comment.strip(), tags_str)
        )
        conn.commit()
        return True
    except Exception:
        return False

def fb_get_all(feature_filter: str = "All", sort: str = "newest") -> list:
    order = {"newest": "created DESC", "oldest": "created ASC",
             "highest": "rating DESC", "lowest": "rating ASC",
             "most liked": "likes DESC"}.get(sort, "created DESC")
    if feature_filter == "All":
        cursor.execute(f"SELECT * FROM feedback ORDER BY {order}")
    else:
        cursor.execute(f"SELECT * FROM feedback WHERE feature=? ORDER BY {order}", (feature_filter,))
    rows = cursor.fetchall()
    cols = ["id","name","feature","rating","comment","tags","likes","created"]
    return [dict(zip(cols, r)) for r in rows]

def fb_get_stats() -> dict:
    cursor.execute("SELECT COUNT(*), AVG(rating) FROM feedback")
    total, avg = cursor.fetchone()
    total = total or 0
    avg   = round(avg or 0, 2)

    cursor.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating ORDER BY rating DESC")
    dist = {r: c for r, c in cursor.fetchall()}

    cursor.execute("""
        SELECT feature, COUNT(*) as cnt, AVG(rating) as avg_r
        FROM feedback GROUP BY feature ORDER BY cnt DESC LIMIT 5
    """)
    top_features = [{"feature": r[0], "count": r[1], "avg": round(r[2], 1)} for r in cursor.fetchall()]

    cursor.execute("""
        SELECT name, rating, comment, created FROM feedback
        ORDER BY rating DESC, created DESC LIMIT 3
    """)
    highlights = [{"name": r[0], "rating": r[1], "comment": r[2], "created": r[3]}
                  for r in cursor.fetchall()]
    return {"total": total, "avg": avg, "dist": dist,
            "top_features": top_features, "highlights": highlights}

def fb_toggle_like(feedback_id: int, session_id: str) -> int:
    try:
        cursor.execute("SELECT 1 FROM feedback_likes WHERE feedback_id=? AND session_id=?",
                       (feedback_id, session_id))
        already = cursor.fetchone()
        if already:
            cursor.execute("DELETE FROM feedback_likes WHERE feedback_id=? AND session_id=?",
                           (feedback_id, session_id))
            cursor.execute("UPDATE feedback SET likes=MAX(0,likes-1) WHERE id=?", (feedback_id,))
        else:
            cursor.execute("INSERT INTO feedback_likes VALUES (?,?)", (feedback_id, session_id))
            cursor.execute("UPDATE feedback SET likes=likes+1 WHERE id=?", (feedback_id,))
        conn.commit()
        cursor.execute("SELECT likes FROM feedback WHERE id=?", (feedback_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0

def fb_delete(feedback_id: int) -> None:
    cursor.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
    cursor.execute("DELETE FROM feedback_likes WHERE feedback_id=?", (feedback_id,))
    conn.commit()

def _star_html(rating: int, size: str = "normal") -> str:
    css = "stars-lg" if size == "large" else "fb-stars"
    filled = "⭐" * rating
    empty  = "☆" * (5 - rating)
    return f'<div class="{css}">{filled}{empty}</div>'

def _session_id() -> str:
    if "_session_uid" not in st.session_state:
        st.session_state._session_uid = str(uuid.uuid4())
    return st.session_state._session_uid

def _liked_set() -> set:
    if "_liked_ids" not in st.session_state:
        st.session_state._liked_ids = set()
    return st.session_state._liked_ids


# =========================================================
# SCREEN: FEEDBACK & COMMUNITY
# =========================================================
def render_feedback_app():
    back_button()
    st.markdown('<div class="screen-header">⭐ Feedback & Community</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">Share your experience · Rate features · Read what others say</div>', unsafe_allow_html=True)

    stats = fb_get_stats()

    st.markdown("<br>", unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    avg_stars = "⭐" * round(stats["avg"]) if stats["avg"] else "—"
    with h1:
        st.markdown(f"""<div class="hero-stat">
            <div class="hero-stat-num">{stats["total"]}</div>
            <div class="hero-stat-lbl">Total Reviews</div></div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""<div class="hero-stat">
            <div class="hero-stat-num">{stats["avg"] or "—"}</div>
            <div class="hero-stat-lbl">Avg Rating / 5</div></div>""", unsafe_allow_html=True)
    with h3:
        dist = stats["dist"]
        five_pct = int(dist.get(5, 0) / max(stats["total"], 1) * 100)
        st.markdown(f"""<div class="hero-stat">
            <div class="hero-stat-num">{five_pct}%</div>
            <div class="hero-stat-lbl">5-Star Reviews</div></div>""", unsafe_allow_html=True)
    with h4:
        top_f = stats["top_features"][0]["feature"].split()[0] if stats["top_features"] else "—"
        st.markdown(f"""<div class="hero-stat">
            <div class="hero-stat-num" style="font-size:1.1rem;">{top_f}</div>
            <div class="hero-stat-lbl">Most Reviewed</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1.9], gap="large")

    with left_col:
        st.markdown("#### 📊 Rating Summary")
        if stats["total"] > 0:
            st.markdown(f"""
            <div style="text-align:center; padding:16px 0;">
                <div class="avg-score">{stats["avg"]}</div>
                <div style="font-size:1.6rem; margin:4px 0;">{"⭐" * round(stats["avg"])}</div>
                <div class="avg-label">Based on {stats["total"]} review{"s" if stats["total"] != 1 else ""}</div>
            </div>""", unsafe_allow_html=True)

            for star in range(5, 0, -1):
                count = stats["dist"].get(star, 0)
                pct   = int(count / max(stats["total"], 1) * 100)
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
                    <span style="font-size:0.75rem; color:#9CA3AF; width:28px; text-align:right;">{star}★</span>
                    <div class="rating-bar-wrap" style="flex:1;">
                        <div class="rating-bar-fill" style="width:{pct}%;"></div>
                    </div>
                    <span style="font-size:0.72rem; color:#4B5563; width:28px;">{count}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No ratings yet. Be the first!")

        st.markdown("<br>", unsafe_allow_html=True)

        if stats["top_features"]:
            st.markdown("#### 🏆 Top Reviewed Features")
            for f in stats["top_features"]:
                stars_mini = "⭐" * round(f["avg"])
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:6px 0; border-bottom:1px solid #1a1a1a;">
                    <span style="font-size:0.78rem; color:#CBD5E1;">{f["feature"]}</span>
                    <span style="font-size:0.72rem; color:#22D3EE;">{stars_mini} {f["avg"]}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### ✍️ Leave a Review")
        with st.form("feedback_form", clear_on_submit=True):
            fb_name    = st.text_input("Your Name", placeholder="e.g. Priya S. (or leave blank for Anonymous)")
            fb_feature = st.selectbox("Which feature are you rating?", FEEDBACK_FEATURES)
            fb_rating  = st.select_slider(
                "Your Rating ⭐",
                options=[1, 2, 3, 4, 5],
                value=5,
                format_func=lambda x: {1:"1 ★ Poor", 2:"2 ★★ Fair", 3:"3 ★★★ Good",
                                        4:"4 ★★★★ Great", 5:"5 ★★★★★ Excellent"}[x]
            )
            fb_comment = st.text_area(
                "Your Comment",
                placeholder="Tell us what worked well, what could be better, or how it helped you…",
                height=100
            )
            fb_tags_chosen = st.multiselect(
                "Quick Tags (optional)",
                FEEDBACK_TAGS,
                max_selections=3,
                help="Pick up to 3 tags that describe your experience"
            )
            submitted = st.form_submit_button("Submit Review ✓", type="primary", use_container_width=True)

        if submitted:
            if not fb_comment.strip():
                st.warning("Please write a comment before submitting.")
            else:
                ok = fb_submit(fb_name, fb_feature, fb_rating, fb_comment, fb_tags_chosen)
                if ok:
                    st.success("🎉 Thank you! Your review has been posted.")
                    st.session_state.feedback_submitted = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Could not save review. Please try again.")

    with right_col:
        st.markdown("#### 💬 Community Reviews")

        fc1, fc2 = st.columns([1.4, 1])
        with fc1:
            filter_feature = st.selectbox(
                "Filter by feature", ["All"] + FEEDBACK_FEATURES,
                key="fb_filter", label_visibility="collapsed"
            )
        with fc2:
            sort_by = st.selectbox(
                "Sort", ["newest", "oldest", "highest", "lowest", "most liked"],
                key="fb_sort", label_visibility="collapsed"
            )

        reviews = fb_get_all(filter_feature, sort_by)
        session_id = _session_id()
        liked_ids  = _liked_set()

        if not reviews:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#4B5563;">
                <div style="font-size:2.5rem; margin-bottom:12px;">💬</div>
                <div style="font-size:0.9rem; color:#6B7280;">No reviews yet for this filter.<br>Be the first to share your experience!</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.caption(f"{len(reviews)} review{'s' if len(reviews) != 1 else ''} found")
            for rev in reviews:
                tags_html = ""
                if rev["tags"]:
                    for tag in rev["tags"].split(","):
                        if tag.strip():
                            tags_html += f'<span class="tag-pill">{tag.strip()}</span>'

                star_filled = "⭐" * rev["rating"]
                star_empty  = "☆" * (5 - rev["rating"])
                is_liked    = rev["id"] in liked_ids

                st.markdown(f"""
                <div class="fb-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <span class="fb-author">{rev["name"]}</span>
                            <span class="fb-feature">{rev["feature"]}</span>
                        </div>
                        <span class="fb-date">{rev["created"][:16]}</span>
                    </div>
                    <div class="fb-stars">{star_filled}{star_empty}</div>
                    <div class="fb-comment">{rev["comment"]}</div>
                    {f'<div style="margin-top:8px;">{tags_html}</div>' if tags_html else ''}
                    <div style="margin-top:10px; display:flex; align-items:center; gap:10px;">
                        <span class="like-badge">{"❤️" if is_liked else "🤍"} {rev["likes"]} helpful</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                btn_label = f"{'❤️ Liked' if is_liked else '🤍 Helpful'} (#{rev['id']})"
                if st.button(btn_label, key=f"like_{rev['id']}", use_container_width=False):
                    new_count = fb_toggle_like(rev["id"], session_id)
                    if is_liked:
                        liked_ids.discard(rev["id"])
                    else:
                        liked_ids.add(rev["id"])
                    st.session_state._liked_ids = liked_ids
                    st.rerun()

        highlights = stats["highlights"]
        if highlights and filter_feature == "All":
            st.markdown("---")
            st.markdown("#### 🌟 Top Reviews")
            for h in highlights:
                star_h = "⭐" * h["rating"]
                st.markdown(f"""
                <div class="fb-card" style="border-left:3px solid #22D3EE;">
                    <div style="display:flex; justify-content:space-between;">
                        <span class="fb-author">{h["name"]}</span>
                        <span class="fb-date">{h["created"][:10]}</span>
                    </div>
                    <div class="fb-stars">{star_h}</div>
                    <div class="fb-comment">"{h["comment"]}"</div>
                </div>""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================
def go_home(): st.session_state.current_screen = "home"
def go_to(screen): st.session_state.current_screen = screen

def back_button():
    col_b, col_rest = st.columns([0.22, 0.78])
    with col_b:
        st.button("Home", on_click=go_home, key=f"back_{st.session_state.current_screen}", use_container_width=True)
    st.divider()


# =========================================================
# URL / ROADMAP HELPERS
# =========================================================
_PLATFORM_EMOJI = {
    "YouTube": "▶️", "GitHub": "🐙", "Udemy": "🎓", "Coursera": "📚",
    "edX": "🎓", "Blog": "✍️", "Generic": "🔗", "Docs": "📄",
}

def _detect_platform(url: str):
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u: return "YouTube", "video"
    if "github.com" in u: return "GitHub", "repo"
    if "udemy.com" in u: return "Udemy", "course"
    if "coursera.org" in u: return "Coursera", "course"
    if "edx.org" in u: return "edX", "course"
    if "docs." in u or "/docs/" in u: return "Docs", "documentation"
    return "Generic", "page"

_PAREN_URL_RE = re.compile(r"([^\n\(]{3,80}?)\s*\(\s*(https?://[^\s\)\],\"\'<>]+[^\s\)\],\"\'<>]*)\s*\)", re.MULTILINE)
_LINE_URL_RE  = re.compile(r"^\s*(?:[-*•\d]+[.):]?\s*)?(https?://[^\s\)\],\"\'<>]+)", re.MULTILINE)
_BARE_URL_RE  = re.compile(r"https?://[^\s\)\],\"\'<>]+", re.MULTILINE)
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def extract_urls_from_roadmap(roadmap_text: str) -> list:
    seen, unique = set(), []
    for m in _PAREN_URL_RE.finditer(roadmap_text):
        label = m.group(1).strip().strip("-•*↓→").strip()
        url   = m.group(2).rstrip(".,;:)")
        if re.fullmatch(r"[\W\d]+", label): label = ""
        if url not in seen:
            seen.add(url); unique.append({"url": url, "label": label})
    for m in _LINE_URL_RE.finditer(roadmap_text):
        url = m.group(1).rstrip(".,;:)")
        if url not in seen:
            seen.add(url); unique.append({"url": url, "label": ""})
    for m in _BARE_URL_RE.finditer(roadmap_text):
        url = m.group(0).rstrip(".,;:)")
        if url not in seen:
            seen.add(url); unique.append({"url": url, "label": ""})
    return unique

def _http_get(url, timeout=8):
    try:
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception: return ""

def _strip_html(html, max_chars=1000):
    s = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[ \t\r\n]+", " ", s).strip()
    return s[:max_chars]

def _title_from_url(url):
    parts = [p for p in urllib.parse.urlparse(url).path.split("/") if p]
    return parts[-1].replace("-"," ").replace("_"," ").title() if parts else url

def _process_single_url(role_name, url, label=""):
    platform, resource_type = _detect_platform(url)
    html  = _http_get(url, timeout=6)
    title = label or _title_from_url(url)
    if html:
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if m: title = re.sub(r"\s+", " ", m.group(1)).strip()[:100]
    system = "You are a career learning advisor. Be concise and practical."
    prompt = (f"You are a career learning advisor for a {role_name} role. "
              f"In exactly 3 sentences, explain why this resource is valuable: "
              f"URL: {url}, Platform: {platform}, Title: {title}")
    summary = _call_groq_direct(
        [{"role": "user", "content": prompt}], system,
        max_tokens=200, temperature=0.3, feature_tag="url_summarizer"
    )
    return {"url": url, "label": label, "platform": platform, "resource_type": resource_type,
            "title": title, "summary": summary}

def _process_url_batch(role_name, batch):
    cards = []
    for item in batch:
        url, label = (item.get("url",""), item.get("label","")) if isinstance(item,dict) else (item,"")
        try: card = _process_single_url(role_name, url, label=label)
        except Exception as exc:
            card = {"url": url, "label": label, "platform": "Unknown", "resource_type": "page",
                    "title": label or _title_from_url(url), "summary": f"Could not summarise. ({type(exc).__name__})"}
        cards.append(card)
    return cards


# =========================================================
# GROQ ASSISTANT HELPERS
# =========================================================
def _build_groq_system_prompt() -> str:
    role_summaries = []
    for rname in role_names[:80]:
        rd = role_data_map.get(rname, {})
        skills_preview = (rd.get("skills", "") or "")[:120].replace("\n", " ")
        role_summaries.append(f"• {rname}: {skills_preview}")

    return f"""You are CareerForge AI — a warm, expert career advisor inside the Smart CareerFordge platform.

YOUR MISSION: Help users of ANY background discover the best tech career path for them.

HOW YOU WORK:
1. If you don't know the user's background, ask 1-2 short friendly questions first.
2. Recommend 2-3 specific roles from the database, explain WHY each fits them personally.
3. For each recommended role, list the top 5 skills to focus on first.
4. Give a realistic learning timeline (beginner / intermediate / job-ready).
5. Keep responses concise but actionable.

TONE: Encouraging, honest, plain language, friendly but professional.

AVAILABLE ROLES ({len(role_names)} total):
{chr(10).join(role_summaries)}

RULES:
• Only recommend roles from the platform database above.
• Never fabricate salaries or statistics.
• Always end with a clear next step.

ANTI-HALLUCINATION RULES (CRITICAL — always follow these):
• NEVER mention a specific company name as a "who is hiring" example — you don't have live data
• NEVER give an exact salary figure — use ranges like "typically 8–14 LPA" and say "approximately"
• NEVER invent a course URL, certification link, or platform link — say "search for X on Coursera" instead
• NEVER recommend a role that is NOT listed in AVAILABLE ROLES above — say "that role isn't in my database, the closest match is [role]"
• NEVER add years of experience as exact numbers — use ranges: "typically 2–4 years"
• If you are uncertain about something, say "I'd recommend verifying this directly on the platform" rather than guessing
• If the user asks about something outside career guidance, say "I'm focused on career advice — I may not be reliable for that topic"
"""

def _call_groq(messages: list, system_prompt: str) -> str:
    return _call_groq_direct(messages, system_prompt,
                              max_tokens=1024, feature_tag="career_assistant")

def _inject_resume_context(messages: list) -> list:
    if not st.session_state.get("resume_text") or not st.session_state.get("resume_skills"):
        return messages
    context = (f"[CANDIDATE RESUME CONTEXT]\n"
               f"Extracted skills: {st.session_state.resume_skills[:600]}\n"
               f"Resume excerpt: {st.session_state.resume_text[:400]}\n[END CONTEXT]")
    enriched = [
        {"role": "user",      "content": context},
        {"role": "assistant", "content": "Got it — I've reviewed your background and skills. How can I help you today?"},
    ]
    return enriched + messages

def _escape_html(text: str) -> str:
    return (text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                .replace('"',"&quot;").replace("\n","<br>"))

def _extract_roles_from_reply(reply: str) -> list:
    found = []
    reply_lower = reply.lower()
    for rname in role_names:
        if rname.lower() in reply_lower and rname not in found:
            found.append(rname)
    return found[:3]

_STARTER_PROMPTS = [
    "I just graduated — what tech role should I target?",
    "I'm from a non-tech background. Where do I start?",
    "I know Python basics. Which AI/ML roles suit me?",
    "What's the difference between Data Engineer and Data Scientist?",
    "I have 2 years of Java experience. What's my best next move?",
    "I want to switch into tech from marketing/finance/healthcare.",
    "Which roles are most in-demand right now in AI?",
    "I want to work in cybersecurity. What should I learn first?",
]
_QUICK_FOLLOWUPS = [
    "Give me a learning roadmap for the top role you suggested",
    "What certifications should I get first?",
    "How long until I'm job-ready?",
    "What portfolio projects should I build?",
]


# =========================================================
# SCREEN: SPLASH
# =========================================================
def render_splash():
    if _LOGO_B64:
        st.markdown(f'<img class="splash-logo" src="data:image/png;base64,{_LOGO_B64}" />', unsafe_allow_html=True)
    else:
        st.markdown('<h1 style="text-align:center;font-size:52px;color:#F3F4F6;'
                    'font-family:Syne,sans-serif;margin-top:80px;">Smart CareerFordge</h1>', unsafe_allow_html=True)
    st.markdown('<div class="splash-tagline">AI-powered career intelligence &nbsp;·&nbsp; Skill matching '
                '&nbsp;·&nbsp; ATS simulation &nbsp;·&nbsp; Live jobs &nbsp;·&nbsp; Salary insights</div>',
                unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1.5, 1, 1.5])
    with col_c:
        if st.button("Enter Platform", type="primary", use_container_width=True):
            go_to("home")


# =========================================================
# SCREEN: HOME
# =========================================================
def render_home():
    st.markdown('<div class="hero-title">Career Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Your AI-powered career companion — powered by LLM & semantic matching</div>', unsafe_allow_html=True)

    APPS = [
        ("Resume Analyzer",      "resume_app"),
        ("Roles Explorer",       "roles_app"),
        ("Skill Match & Gap",    "skill_match_app"),
        ("Top Role Picks",       "top_roles_app"),
        ("Career Transitions",   "transition_app"),
        ("ATS Analysis",         "ats_app"),
        ("Readiness Dash",       "readiness_app"),
        ("Career Assistant",     "assistant_app"),
        ("Job Listings",         "jobs_app"),
        ("Job Market Insights",  "market_app"),
        ("Salary Trends",        "salary_chart_app"),
        ("AI Cover Letter",      "coverletter_app"),
        ("Salary Estimator",     "salary_est_app"),
        ("LinkedIn Bio Writer",  "linkedin_app"),
        ("Feedback & Reviews",   "feedback_app"),
        ("About / Help",         "about_app"),
    ]

    cols_per_row = 4
    for row_start in range(0, len(APPS), cols_per_row):
        row_apps = APPS[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="small")
        for col, (label, screen) in zip(cols, row_apps):
            with col:
                if st.button(label, key=f"tile_{screen}", use_container_width=True):
                    go_to(screen)

    st.markdown("<br>", unsafe_allow_html=True)
    has_resume = bool(st.session_state.resume_text)
    has_jd     = bool(st.session_state.jd_text)
    applied_count = len(st.session_state.get("applied_jobs", []))
    st.caption(
        f"{'✓' if has_resume else '○'} Resume loaded  |  "
        f"{'✓' if has_jd else '○'} JD loaded  |  "
        f"{len(role_names)} roles  |  "
        f"{len(STRONG_TECH_KEYWORDS)} skills indexed  |  "
        f"{applied_count} jobs applied"
    )

    ls_active  = _ls_enabled()
    ls_project = os.getenv("LANGCHAIN_PROJECT", "Smart CareerFordge")

    if ls_active:
        st.markdown(f"""
        <div style="background:#080f18; border:1px solid #1a2535; border-radius:12px;
                    padding:12px 18px; margin-top:10px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.78rem; color:#22D3EE;">🟢 LangSmith Tracing: Active</span><br>
                <span style="font-size:0.72rem; color:#4B5563;">Project: {ls_project}</span>
            </div>
            <div style="font-size:0.72rem; color:#22D3EE;">Monitoring all LLM calls →</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#080f18; border:1px solid #1a2535; border-radius:12px;
                    padding:12px 18px; margin-top:10px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.78rem; color:#4B5563;">⚫ LangSmith Tracing: Off</span>
            </div>
            <div style="font-size:0.72rem; color:#4B5563;">Add LANGCHAIN keys to .env to enable</div>
        </div>""", unsafe_allow_html=True)

    ntfy_topic = os.getenv("NTFY_TOPIC", "")
    if ntfy_topic:
        st.markdown(f"""
        <div style="background:#080f18; border:1px solid #1a2535; border-radius:12px;
                    padding:12px 18px; margin-top:8px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.78rem; color:#22c55e;">🔔 Push Alerts: Active</span><br>
                <span style="font-size:0.72rem; color:#4B5563;">Topic: {ntfy_topic}</span>
            </div>
            <div style="font-size:0.72rem; color:#22c55e;">Rate limits · Errors · Slow responses →</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#080f18; border:1px solid #1a2535; border-radius:12px;
                    padding:12px 18px; margin-top:8px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.78rem; color:#4B5563;">🔕 Push Alerts: Off</span>
            </div>
            <div style="font-size:0.72rem; color:#4B5563;">Add NTFY_TOPIC to .env to enable free alerts</div>
        </div>""", unsafe_allow_html=True)

    stats = fb_get_stats()
    if stats["total"] > 0:
        avg_stars_home = "⭐" * round(stats["avg"])
        st.markdown(f"""
        <div style="background:#080f18; border:1px solid #1a2535; border-radius:12px;
                    padding:14px 18px; margin-top:16px; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.82rem; color:#9CA3AF;">Community Rating</span><br>
                <span style="font-size:1.3rem;">{avg_stars_home}</span>
                <span style="font-size:0.9rem; color:#22D3EE; font-weight:700; margin-left:6px;">{stats["avg"]} / 5</span>
                <span style="font-size:0.75rem; color:#4B5563; margin-left:8px;">from {stats["total"]} reviews</span>
            </div>
            <div style="font-size:0.78rem; color:#22D3EE; cursor:pointer;">
                → View all reviews
            </div>
        </div>""", unsafe_allow_html=True)


# =========================================================
# SCREEN: RESUME ANALYZER
# =========================================================
def render_resume_app():
    back_button()
    st.markdown('<div class="screen-header">Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">Upload your resume — get skills, score card & quality metrics</div>', unsafe_allow_html=True)

    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume_uploader")
    if resume_file:
        os.makedirs("uploads", exist_ok=True)
        resume_path = os.path.join("uploads", resume_file.name)
        with open(resume_path, "wb") as f:
            f.write(resume_file.read())
        full_text, resume_skills_text = process_resume(resume_path)
        st.session_state.resume_text   = full_text
        st.session_state.resume_skills = resume_skills_text
        score_card = compute_resume_score_card(full_text, resume_skills_text)
        st.session_state.resume_score_card = score_card
        cursor.execute("INSERT INTO resume_data (filename, skills, full_text) VALUES (?,?,?)",
                       (resume_file.name, resume_skills_text, full_text))
        conn.commit()
        st.success("Resume processed successfully!")

    if st.session_state.resume_skills:
        sc = st.session_state.get("resume_score_card")
        if sc is None:
            sc = compute_resume_score_card(st.session_state.resume_text, st.session_state.resume_skills)
            st.session_state.resume_score_card = sc

        st.markdown("### 📊 Resume Score Card")
        grade_color = {"A": "#22c55e", "B": "#60a5fa", "C": "#f59e0b", "D": "#ef4444"}.get(sc["grade"], "#9CA3AF")
        st.markdown(f"""
        <div class="{sc['css']}">
            <div class="score-number">{sc['total']}</div>
            <div class="score-grade" style="color:{grade_color}">Grade {sc['grade']}</div>
            <div class="score-label">{sc['label']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Score Breakdown**")
        for dim, (earned, total) in sc["breakdown"].items():
            pct = int(earned / total * 100)
            col_label, col_bar, col_val = st.columns([2.5, 5, 1])
            with col_label: st.caption(dim)
            with col_bar:   st.progress(pct / 100)
            with col_val:   st.caption(f"{earned}/{total}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Extracted Skills", "Skill Quality Metrics", "Raw Resume Text"])
        with tab1:
            st.markdown("**Parsed Skills**")
            st.text(st.session_state.resume_skills)
        with tab2:
            density, kw_hits = compute_section1_scores(st.session_state.resume_skills)
            c1, c2 = st.columns(2)
            with c1:
                _badge("Skills Density", density, 90, 70, fmt="{:.1f}%",
                       caption="unique skills / total listed items x 100  |  ideal >= 90%")
            with c2:
                _badge("Tech Keyword Score", kw_hits, 20, 10, fmt="{} keywords",
                       caption=f"matched from {len(STRONG_TECH_KEYWORDS)} skills  |  ideal >= 20")
        with tab3:
            st.text(st.session_state.resume_text[:3000] + ("..." if len(st.session_state.resume_text) > 3000 else ""))
    else:
        st.info("Upload a resume PDF above to get started.")


# =========================================================
# SCREEN: ROLES EXPLORER
# =========================================================
def render_roles_app():
    back_button()
    st.markdown('<div class="screen-header">Career Roles Explorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="screen-subheader">{len(role_names)} roles loaded from Career_Path.pdf</div>', unsafe_allow_html=True)

    browse_option = st.radio("Input method", ["Select Role", "Search by Query"], horizontal=True)

    def _render_role_detail(role_name, rdata):
        quality = compute_section2_scores(role_name, rdata, embeddings)
        tab_ov, tab_sk, tab_rm, tab_ce = st.tabs(["Overview", "Skills", "Roadmap", "Certifications"])
        with tab_ov:
            _badge("Data Quality", quality, 80, 50, fmt="{}%")
        with tab_sk:
            st.text(rdata["skills"])
        with tab_rm:
            st.markdown("#### Raw Roadmap")
            st.text(rdata["learning_roadmap"])
            st.divider()
            st.markdown("#### Resource Summaries")
            roadmap_text = rdata.get("learning_roadmap", "")
            if not roadmap_text or roadmap_text == "Not listed":
                st.info("No roadmap data available for this role.")
            else:
                urls_found = extract_urls_from_roadmap(roadmap_text)
                if not urls_found:
                    st.info("No resource links found in this roadmap.")
                else:
                    total_urls = len(urls_found)
                    cache_key  = f"roadmap_cards__{role_name}"
                    if cache_key in st.session_state:
                        for card in st.session_state[cache_key]:
                            emoji = _PLATFORM_EMOJI.get(card["platform"], "🔗")
                            with st.expander(f"{emoji} {card.get('title','Resource')}", expanded=True):
                                st.markdown(f"**[{card.get('title','Open')}]({card['url']})**")
                                st.write(card["summary"])
                    else:
                        all_cards = []
                        for batch_start in range(0, total_urls, 3):
                            batch = urls_found[batch_start:batch_start+3]
                            new_cards = _process_url_batch(role_name, batch)
                            all_cards.extend(new_cards)
                        st.session_state[cache_key] = all_cards
                        st.rerun()
        with tab_ce:
            st.text(rdata["certifications"])

    if browse_option == "Select Role":
        selected_role = st.selectbox("Pick a role", sorted(role_names))
        if st.button("Show Details", type="primary"):
            rdata = get_role_data(selected_role)
            if rdata: _render_role_detail(selected_role, rdata)
    else:
        user_query = st.text_input("Enter a role name or keyword")
        if st.button("Search", type="primary"):
            docs = role_vectordb.similarity_search(user_query, k=1)
            if docs:
                meta_role = docs[0].metadata.get("role", "")
                if meta_role:
                    st.info(f"Best match: **{meta_role}**")
                    rdata = get_role_data(meta_role)
                    if rdata: _render_role_detail(meta_role, rdata)


# =========================================================
# SCREEN: SKILL MATCH
# =========================================================
def render_skill_match_app():
    back_button()
    st.markdown('<div class="screen-header">Skill Match & Gap Analysis</div>', unsafe_allow_html=True)
    if not st.session_state.resume_skills:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")
        return
    selected_role_match = st.selectbox("Select a role to compare", sorted(role_names))
    if st.button("Run Match Analysis", type="primary"):
        rdata = get_role_data(selected_role_match)
        if rdata:
            matched, missing, score = match_skills(rdata["skills"], st.session_state.resume_skills, embeddings)
            lbl   = "Strong Match" if score >= 75 else ("Partial Match" if score >= 50 else "Skill Gap")
            total = len(matched) + len(missing)
            col_s, col_c = st.columns([1, 2])
            with col_s: st.metric(lbl, f"{score}%")
            with col_c:
                st.write(f"**{len(matched)}** of **{total}** required skills matched")
                st.progress(score / 100)
            jaccard, sem_score, precision, recall = compute_section3_scores(
                rdata["skills"], st.session_state.resume_skills, matched, missing, embeddings)
            st.markdown("#### Advanced Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1: _badge("Jaccard", jaccard, 0.45, 0.25, fmt="{:.2f}")
            with m2: _badge("Semantic", sem_score, 0.70, 0.55, fmt="{:.2f}")
            with m3: _badge("Precision", precision, 50, 30, fmt="{:.0f}%")
            with m4: _badge("Recall", recall, 70, 50, fmt="{:.0f}%")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Skills You Have**")
                for sk in sorted(set(matched)): st.markdown(f"- {sk}")
            with col2:
                st.markdown("**Skills to Acquire**")
                if missing:
                    for sk in sorted(set(missing)): st.markdown(f"- {sk}")
                else: st.success("You have all required skills!")


# =========================================================
# SCREEN: TOP ROLES
# =========================================================
def render_top_roles_app():
    back_button()
    st.markdown('<div class="screen-header">Top Role Suggestions</div>', unsafe_allow_html=True)
    if not st.session_state.resume_skills:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")
        return
    if st.button("Find My Top Roles", type="primary"):
        with st.spinner("Scoring all roles..."):
            top_roles = suggest_top_3_roles(st.session_state.resume_skills, top_k=3)
        if top_roles:
            top_jaccard, median_cosine = compute_section4_scores(top_roles, st.session_state.resume_skills, embeddings)
            st.markdown("#### Suggestion Quality Summary")
            sq1, sq2 = st.columns(2)
            with sq1: _badge("Top Match Jaccard", top_jaccard, 0.35, 0.20, fmt="{:.2f}")
            with sq2: _badge("Median Similarity", median_cosine, 0.70, 0.55, fmt="{:.2f}")
            st.divider()
            for i, (role, cascade_pct, combined, matched, missing) in enumerate(top_roles, start=1):
                st.markdown(f"### {i}. {role}")
                ca, cb = st.columns([1, 2])
                with ca:
                    badge = "Strong Fit" if cascade_pct >= 75 else ("Partial Fit" if cascade_pct >= 50 else "Skill Gap")
                    st.metric(badge, f"{cascade_pct}%")
                with cb:
                    total = len(matched) + len(missing)
                    st.write(f"**{len(matched)}** of **{total}** role skills matched")
                    st.progress(cascade_pct / 100)
                with st.expander(f"Skill breakdown — {role}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**You Have**")
                        for sk in sorted(set(matched)): st.markdown(f"- {sk}")
                    with c2:
                        st.markdown("**To Learn**")
                        for sk in sorted(set(missing)): st.markdown(f"- {sk}")


# =========================================================
# SCREEN: CAREER TRANSITIONS
# =========================================================
def render_transition_app():
    back_button()
    st.markdown('<div class="screen-header">Career Transitions</div>', unsafe_allow_html=True)
    if not st.session_state.resume_skills:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")
        return
    current_role = st.selectbox("Your Current Role", sorted(role_names))
    if st.button("Find Best Transitions", type="primary"):
        with st.spinner("Analyzing transitions..."):
            transitions = resume_based_transitions(st.session_state.resume_skills, current_role, top_k=3)
        if not transitions:
            st.warning("Not enough resume skill data to compute transitions.")
            return
        gap, missing_load, realistic_overlap = compute_section6_scores(transitions, current_role, st.session_state.resume_skills)
        st.markdown("#### Transition Risk Overview")
        r1, r2, r3 = st.columns(3)
        with r1: _badge("Transition Gap", gap, 30, 55, fmt="{:.1f}%", reverse=True)
        with r2: _badge("Missing Skills Load", missing_load, 4, 8, fmt="{:.1f} skills", reverse=True)
        with r3: _badge("Realistic Overlap", realistic_overlap, 55, 35, fmt="{:.1f}%")
        st.divider()
        for i, t in enumerate(transitions, start=1):
            st.markdown(f"## {i}. {current_role} → {t['target_role']}")
            pc1, pc2, pc3 = st.columns(3)
            with pc1: st.metric("Skill Match", f"{t['match_percent']}%")
            with pc2: st.metric("Transition Gap", f"{100 - t['match_percent']}%")
            with pc3: st.metric("Missing Skills", f"{len(t['missing_skills'])}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Skills You Have**")
                st.write(", ".join(t["matched_skills"]) or "—")
            with col2:
                st.markdown("**Skills to Learn**")
                st.write(", ".join(t["missing_skills"]) or "—")
            st.markdown("**Why this transition fits you**")
            with st.spinner("Generating insights..."):
                st.write(llm_transition_reason(
                    st.session_state.resume_text, current_role,
                    t["target_role"], match_pct=t["match_percent"]
                ))
            st.markdown("**Personalized Learning Roadmap**")
            with st.spinner("Building roadmap..."):
                st.write(llm_learning_roadmap(t["target_role"], t["missing_skills"]))
            st.divider()


# =========================================================
# SCREEN: ATS ANALYSIS
# =========================================================
def render_ats_app():
    back_button()
    st.markdown('<div class="screen-header">ATS Analysis</div>', unsafe_allow_html=True)
    if not st.session_state.resume_text:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")
    jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd_uploader")
    if jd_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(jd_file.read()); jd_path = tmp.name
        loader  = PyPDFLoader(jd_path)
        jd_text = "\n".join(p.page_content for p in loader.load())
        st.session_state.jd_text = jd_text
        st.success("JD uploaded successfully")
    if st.session_state.resume_text and st.session_state.jd_text:
        if st.button("Run ATS Analysis", type="primary"):
            with st.spinner("Analyzing resume against JD..."):
                ats_text = llm_ats_analysis(st.session_state.resume_text, st.session_state.jd_text)
                st.session_state.ats_result = ats_text
        if st.session_state.ats_result:
            tab_report, tab_metrics = st.tabs(["ATS Report", "Semantic Alignment"])
            with tab_report:
                st.text(st.session_state.ats_result)
            with tab_metrics:
                sem_match, jd_coverage = compute_section5_scores(
                    st.session_state.resume_text, st.session_state.jd_text,
                    st.session_state.resume_skills, embeddings)
                a1, a2 = st.columns(2)
                with a1: _badge("Text Semantic Match", sem_match, 0.65, 0.45, fmt="{:.2f}")
                with a2: _badge("JD Skill Coverage", jd_coverage, 60, 35,
                                fmt="{}%" if jd_coverage is not None else "{}")
    elif not st.session_state.jd_text:
        st.info("Upload a JD PDF above to proceed.")


# =========================================================
# SCREEN: READINESS DASHBOARD
# =========================================================
def render_readiness_app():
    back_button()
    st.markdown('<div class="screen-header">Readiness Dashboard</div>', unsafe_allow_html=True)
    if not st.session_state.resume_text:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")
        return
    readiness = compute_resume_readiness(st.session_state.resume_text, st.session_state.resume_skills)
    labels, values = list(readiness.keys()), list(readiness.values())
    st.markdown("#### Component Scores")
    cols = st.columns(3)
    for i, (label, val) in enumerate(readiness.items()):
        with cols[i % 3]:
            status = "Good" if val >= 70 else ("Fair" if val >= 40 else "Low")
            st.metric(f"[{status}] {label}", f"{val}%")
    st.divider()
    st.markdown("#### Readiness Radar")
    labels_plot = labels + [labels[0]]
    values_plot = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels_plot))
    fig = plt.figure(figsize=(5, 5), facecolor="#000000")
    ax  = plt.axes(polar=True, facecolor="#111111")
    ax.plot(angles, values_plot, color="#0ea5c9", linewidth=2.5)
    ax.fill(angles, values_plot, alpha=0.20, color="#0ea5c9")
    ax.plot(angles, [100] * len(angles), color="#2b2b2b", linewidth=0.8, linestyle="--")
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=8, color="#b0b8c8")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#707880")
    ax.spines["polar"].set_color("#2b2b2b")
    ax.yaxis.set_tick_params(labelcolor="#707880")
    ax.grid(color="#2b2b2b", linewidth=0.7)
    ax.set_title("Career Readiness", pad=18, color="#b0b8c8", fontsize=12, fontweight="bold")
    st.pyplot(fig)
    st.divider()
    st.markdown("#### Export Career Intelligence Report")
    candidate_name = st.text_input("Your Name (for report)", value=st.session_state.get("candidate_name", ""))
    st.session_state.candidate_name = candidate_name
    if st.session_state.ats_result and candidate_name and st.session_state.resume_skills:
        _top3_raw        = suggest_top_3_roles(st.session_state.resume_skills)
        _top3_for_report = [(role, pct / 100) for role, pct, *_ in _top3_raw]
        current_role     = sorted(role_names)[0]
        report_text = generate_career_report(
            candidate_name, st.session_state.resume_skills, st.session_state.ats_result,
            _top3_for_report, resume_based_transitions(st.session_state.resume_skills, current_role), current_role)
        st.download_button("Download Career Report (.txt)", data=report_text,
                           file_name="career_intelligence_report.txt", mime="text/plain")
    else:
        st.info("Run ATS Analysis and enter your name to enable report download.")


# =========================================================
# SCREEN: JOB MARKET INSIGHTS
# =========================================================
def render_market_app():
    back_button()
    st.markdown('<div class="screen-header">Job Market Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">Live job counts per role from Adzuna India — updated in real time</div>', unsafe_allow_html=True)

    DEFAULT_ROLES = [
        "Data Analyst", "Python Developer", "Machine Learning Engineer",
        "DevOps Engineer", "Full Stack Developer", "Data Scientist",
        "Cloud Engineer", "Cybersecurity Analyst",
    ]

    if st.session_state.resume_skills:
        if st.button("Use My Top Roles from Resume", type="primary"):
            with st.spinner("Computing your top roles..."):
                top = suggest_top_3_roles(st.session_state.resume_skills, top_k=5)
                st.session_state["_insight_roles"] = [r[0] for r in top]

    roles_to_check = st.session_state.get("_insight_roles", DEFAULT_ROLES)

    role_names_set   = set(role_names)
    safe_default     = [r for r in roles_to_check if r in role_names_set][:6]
    if not safe_default:
        safe_default = sorted(role_names)[:6]

    st.markdown('<p class="section-hint">Or pick roles manually:</p>', unsafe_allow_html=True)
    selected_roles = st.multiselect(
        "Roles to analyse", sorted(role_names),
        default=safe_default,
        key="market_roles_select"
    )
    if selected_roles:
        roles_to_check = selected_roles

    if st.button("🔍 Fetch Live Job Counts", type="primary"):
        with st.spinner(f"Querying Adzuna for {len(roles_to_check)} roles…"):
            insights = fetch_market_insights(roles_to_check)
        st.session_state.market_insights = insights

    insights = st.session_state.get("market_insights", {})
    if insights:
        st.markdown("---")
        st.markdown("#### 📊 Live Openings in India")

        sorted_insights = sorted(insights.items(), key=lambda x: x[1], reverse=True)
        cols = st.columns(4)
        for i, (role, count) in enumerate(sorted_insights):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-number">{count:,}</div>
                    <div class="insight-label">{role}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        st.markdown("#### Bar Chart — Job Openings")
        if sorted_insights:
            fig, ax = plt.subplots(figsize=(9, 4), facecolor="#000000")
            ax.set_facecolor("#0a0a0a")
            roles_list  = [r for r, _ in sorted_insights]
            counts_list = [c for _, c in sorted_insights]
            colors = ["#22D3EE" if i == 0 else "#1d4ed8" for i in range(len(roles_list))]
            bars = ax.barh(roles_list[::-1], counts_list[::-1], color=colors[::-1],
                           height=0.55, edgecolor="#1f1f1f")
            for bar, count in zip(bars, counts_list[::-1]):
                ax.text(bar.get_width() + max(counts_list) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{count:,}", va="center", ha="left",
                        color="#9CA3AF", fontsize=8)
            ax.set_xlabel("Open Positions", color="#9CA3AF", fontsize=9)
            ax.tick_params(colors="#9CA3AF", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#1f1f1f")
            ax.spines["bottom"].set_color("#1f1f1f")
            ax.set_title("Live Job Openings — India (Adzuna)", color="#F3F4F6", fontsize=11, pad=12)
            plt.tight_layout()
            st.pyplot(fig)

        top_role, top_count = sorted_insights[0]
        st.success(f"🔥 **Hottest role right now:** {top_role} with **{top_count:,}** open positions in India!")
        st.caption("Powered by Adzuna Jobs API · Counts reflect available listings at time of query")
    else:
        st.info("Click 'Fetch Live Job Counts' to load real-time market data from Adzuna.")


# =========================================================
# SCREEN: SALARY TRENDS CHART
# =========================================================
def render_salary_chart_app():
    back_button()
    st.markdown('<div class="screen-header">Salary Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">Average salary comparison across your top role recommendations (India market)</div>', unsafe_allow_html=True)

    if not st.session_state.resume_skills:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")

    selected_roles_sal = st.multiselect(
        "Select roles to compare (or auto-detect from resume)",
        sorted(role_names), key="salary_roles_select",
        help="Leave empty to auto-detect from your resume"
    )

    if st.button("📈 Generate Salary Chart", type="primary"):
        with st.spinner("Fetching salary data..."):
            if selected_roles_sal:
                top_roles_data = [(r, 75, 75, [], []) for r in selected_roles_sal[:5]]
            elif st.session_state.resume_skills:
                top_roles_data = suggest_top_3_roles(st.session_state.resume_skills, top_k=5)
            else:
                st.warning("Please load a resume or select roles manually.")
                return
            salary_data = compute_salary_trends(top_roles_data, st.session_state.resume_skills)
        st.session_state.salary_chart_data = salary_data

    salary_data = st.session_state.get("salary_chart_data", {})
    if salary_data:
        st.markdown("---")
        sorted_sal = sorted(salary_data.items(), key=lambda x: x[1], reverse=True)
        st.markdown("#### 💰 Estimated Average CTC (LPA)")
        cols = st.columns(min(len(sorted_sal), 5))
        for i, (role, lpa) in enumerate(sorted_sal):
            with cols[i % len(cols)]:
                st.metric(role.split(" ")[-1] if len(role) > 20 else role, f"₹{lpa} LPA")

        st.divider()
        st.markdown("#### Salary Comparison Bar Chart")
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#000000")
        ax.set_facecolor("#0a0a0a")
        roles_list = [r for r, _ in sorted_sal]
        lpa_list   = [v for _, v in sorted_sal]

        bar_colors = []
        for i in range(len(roles_list)):
            ratio = 1 - i / max(len(roles_list) - 1, 1)
            r_val = int(0x22 + (0x1d - 0x22) * ratio)
            g_val = int(0xD3 + (0x4e - 0xD3) * ratio)
            b_val = int(0xEE + (0xd8 - 0xEE) * ratio)
            bar_colors.append(f"#{r_val:02x}{g_val:02x}{b_val:02x}")

        bars = ax.bar(range(len(roles_list)), lpa_list, color=bar_colors,
                      width=0.55, edgecolor="#1f1f1f")
        for bar, val in zip(bars, lpa_list):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"₹{val}L", ha="center", va="bottom", color="#9CA3AF", fontsize=8)

        ax.set_xticks(range(len(roles_list)))
        ax.set_xticklabels([r[:18] for r in roles_list], rotation=20, ha="right",
                           color="#9CA3AF", fontsize=8)
        ax.set_ylabel("LPA (Lakhs Per Annum)", color="#9CA3AF", fontsize=9)
        ax.tick_params(axis="y", colors="#9CA3AF", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#1f1f1f")
        ax.spines["bottom"].set_color("#1f1f1f")
        ax.set_title("Salary Comparison — India Tech Market", color="#F3F4F6", fontsize=11, pad=12)
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()
        st.markdown("#### Detailed Breakdown")
        for role, lpa in sorted_sal:
            with st.expander(f"💼 {role} — ₹{lpa} LPA average"):
                low_est  = round(lpa * 0.75, 1)
                high_est = round(lpa * 1.35, 1)
                st.markdown(f"**Estimated range:** ₹{low_est} – ₹{high_est} LPA")
                st.markdown(f"**Fresher entry:** ₹{round(lpa * 0.55, 1)} – ₹{round(lpa * 0.75, 1)} LPA")
                st.markdown(f"**3-5 years exp:** ₹{round(lpa * 0.90, 1)} – ₹{round(lpa * 1.25, 1)} LPA")
                st.markdown(f"**5+ years / senior:** ₹{round(lpa * 1.25, 1)} – ₹{round(lpa * 1.8, 1)} LPA")

        st.caption("⚠️ Salary estimates based on Adzuna listings + Indian market benchmarks (2024–25). Actual offers vary by company, city, and negotiation.")
    else:
        st.info("Click 'Generate Salary Chart' to compare salaries across roles.")


# =========================================================
# SCREEN: AI COVER LETTER GENERATOR
# =========================================================
def render_coverletter_app():
    back_button()
    st.markdown('<div class="screen-header">AI Cover Letter Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">One-click personalised cover letter using your resume + JD + Groq AI</div>', unsafe_allow_html=True)

    has_resume = bool(st.session_state.resume_text)
    has_jd     = bool(st.session_state.jd_text)

    if not has_resume:
        st.warning("⚠️ No resume loaded. Upload in Resume Analyzer for best results.")
    if not has_jd:
        st.info("ℹ️ No JD loaded. Go to ATS Analysis to upload a Job Description first, or type one below.")

    jd_input = st.text_area(
        "Job Description (paste here or load via ATS Analysis)",
        value=st.session_state.jd_text[:1500] if st.session_state.jd_text else "",
        height=140, key="cl_jd_input",
        placeholder="Paste the job description here…"
    )
    if jd_input.strip():
        st.session_state.jd_text = jd_input.strip()

    col1, col2 = st.columns(2)
    with col1:
        cl_name = st.text_input("Your Name", value=st.session_state.get("candidate_name", ""),
                                 key="cl_name")
    with col2:
        cl_location = st.text_input("Your Location", value="India", key="cl_location")

    exp_level = st.session_state.get("user_experience_level", "fresher")
    exp_label = {"fresher":"Fresher","junior":"Junior","mid":"Mid-Level","senior":"Senior"}.get(exp_level,"Professional")

    if st.button("✍️ Generate Cover Letter", type="primary"):
        if not st.session_state.resume_text:
            st.warning("Please upload a resume first via Resume Analyzer.")
        elif not st.session_state.jd_text:
            st.warning("Please paste or upload a job description above.")
        else:
            with st.spinner("Generating personalised cover letter with Groq AI…"):
                cover_letter = generate_cover_letter(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                    cl_name or "Applicant",
                    exp_label,
                    cl_location
                )
            st.session_state.cover_letter = cover_letter
            st.session_state.candidate_name = cl_name

    if st.session_state.get("cover_letter"):
        st.markdown("---")
        st.markdown("#### 📄 Your Cover Letter")
        st.markdown(f"""
        <div style="background:#080f18; border:1px solid #1a2535; border-left:4px solid #22D3EE;
                    border-radius:12px; padding:24px; font-size:0.9rem; line-height:1.75;
                    color:#E6E6E6; white-space:pre-wrap;">
{st.session_state.cover_letter}
        </div>
        """, unsafe_allow_html=True)

        col_copy, col_dl = st.columns(2)
        with col_dl:
            st.download_button(
                "📥 Download Cover Letter (.txt)",
                data=st.session_state.cover_letter,
                file_name="cover_letter.txt", mime="text/plain"
            )
        with col_copy:
            if st.button("🔄 Regenerate", key="regen_cl"):
                st.session_state.cover_letter = ""
                st.rerun()

        st.caption("💡 Tip: Customise the letter further by tweaking key sentences for each application.")


# =========================================================
# SCREEN: SALARY ESTIMATOR
# =========================================================
def render_salary_est_app():
    back_button()
    st.markdown('<div class="screen-header">Salary Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">AI-powered salary range estimate based on your skills, location & experience</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        target_role_sal = st.selectbox("Target Role", sorted(role_names), key="sal_role")
        sal_location    = st.text_input("Location", value="Bangalore", key="sal_location")
    with col2:
        sal_exp_years = st.number_input("Years of Experience", min_value=0, max_value=30, value=0, key="sal_exp")
        sal_exp_level = st.selectbox(
            "Experience Level",
            ["fresher", "junior", "mid", "senior"],
            format_func=lambda x: {"fresher":"🌱 Fresher","junior":"🔰 Junior","mid":"⚡ Mid-Level","senior":"🚀 Senior"}[x],
            key="sal_exp_level"
        )

    sal_skills = ""
    if st.session_state.resume_skills:
        st.info(f"Using skills from your resume: {st.session_state.resume_skills[:80]}…")
        sal_skills = st.session_state.resume_skills
    else:
        sal_skills = st.text_area("Enter your skills (comma separated)",
                                   placeholder="Python, Machine Learning, SQL, AWS…",
                                   height=80, key="sal_skills_input")

    if st.button("💰 Estimate My Salary", type="primary"):
        if not sal_skills.strip():
            st.warning("Please enter your skills or upload a resume first.")
        else:
            with st.spinner("Analysing market rates with Groq AI…"):
                estimate = estimate_salary(
                    sal_skills, sal_location, int(sal_exp_years),
                    sal_exp_level, target_role_sal
                )
            st.session_state.salary_estimate = estimate

    if st.session_state.get("salary_estimate"):
        st.markdown("---")
        st.markdown("#### 💰 Salary Estimate Report")
        st.markdown(f"""
        <div style="background:#080f18; border:1px solid #1a2535; border-left:4px solid #22D3EE;
                    border-radius:12px; padding:20px; font-size:0.88rem; line-height:1.75; color:#E6E6E6;">
{st.session_state.salary_estimate}
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "📥 Download Salary Report",
            data=st.session_state.salary_estimate,
            file_name="salary_estimate.txt", mime="text/plain"
        )
        st.caption("⚠️ Estimates are AI-generated based on market knowledge. Actual offers depend on company, negotiation, and candidate performance.")

        if st.button("🔄 Re-estimate", key="regen_sal"):
            st.session_state.salary_estimate = ""
            st.rerun()


# =========================================================
# SCREEN: LINKEDIN BIO WRITER
# =========================================================
def render_linkedin_app():
    back_button()
    st.markdown('<div class="screen-header">LinkedIn Bio Writer</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">AI rewrites your resume skills into a compelling LinkedIn About section</div>', unsafe_allow_html=True)

    if not st.session_state.resume_text:
        st.warning("No resume loaded. Please go to Resume Analyzer first.")

    col1, col2 = st.columns(2)
    with col1:
        li_name = st.text_input("Your Name", value=st.session_state.get("candidate_name", ""), key="li_name")
        li_exp_level = st.selectbox(
            "Experience Level",
            ["fresher","junior","mid","senior"],
            format_func=lambda x: {"fresher":"🌱 Fresher","junior":"🔰 Junior","mid":"⚡ Mid-Level","senior":"🚀 Senior"}[x],
            key="li_exp"
        )
    with col2:
        li_target_role = st.selectbox("Target Role / Headline", sorted(role_names), key="li_role")

    li_skills_input = st.text_area(
        "Skills (auto-filled from resume, or edit manually)",
        value=st.session_state.resume_skills[:500] if st.session_state.resume_skills else "",
        height=80, key="li_skills"
    )

    if st.button("✨ Generate LinkedIn Bio", type="primary"):
        resume_text  = st.session_state.resume_text or "No resume uploaded"
        skills_input = li_skills_input or st.session_state.resume_skills or "No skills provided"
        with st.spinner("Crafting your LinkedIn bio with Groq AI…"):
            bio = generate_linkedin_bio(resume_text, skills_input, li_exp_level, li_target_role, li_name)
        st.session_state.linkedin_bio = bio
        st.session_state.candidate_name = li_name

    if st.session_state.get("linkedin_bio"):
        st.markdown("---")
        st.markdown("#### 💼 Your LinkedIn About Section")

        char_count = len(st.session_state.linkedin_bio)
        count_color = "#22c55e" if char_count <= 2600 else "#ef4444"
        st.markdown(f'<p style="font-size:0.75rem; color:{count_color}; text-align:right;">'
                    f'{char_count} characters (LinkedIn max: 2,600)</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#080f18; border:1px solid #1a2535; border-left:4px solid #22D3EE;
                    border-radius:12px; padding:20px; font-size:0.9rem; line-height:1.75;
                    color:#E6E6E6; white-space:pre-wrap;">
{st.session_state.linkedin_bio}
        </div>
        """, unsafe_allow_html=True)

        col_dl, col_regen = st.columns(2)
        with col_dl:
            st.download_button(
                "📥 Download Bio (.txt)",
                data=st.session_state.linkedin_bio,
                file_name="linkedin_about.txt", mime="text/plain"
            )
        with col_regen:
            if st.button("🔄 Regenerate Bio", key="regen_li"):
                st.session_state.linkedin_bio = ""
                st.rerun()

        st.caption("💡 Tips: Copy this into your LinkedIn profile → Edit → About section. Customise the tone to match your personality.")

        with st.expander("📋 How to use on LinkedIn"):
            st.markdown("""
1. Go to **linkedin.com** → click your profile picture
2. Click the **✏️ Edit** pencil near your photo
3. Scroll to **About** → click **Edit**
4. Paste this text and personalise any details
5. LinkedIn SEO tip: make sure key skills appear in the first 2 lines (shown before "See more")
            """)


# =========================================================
# SCREEN: CAREER ASSISTANT
# =========================================================
def render_assistant_app():
    back_button()
    st.markdown('<div class="screen-header">Career Guidance Assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="screen-subheader">AI career advisor · Groq Llama-3.3-70B · Adzuna live jobs · {len(role_names)} roles</div>', unsafe_allow_html=True)

    has_resume = bool(st.session_state.get("resume_text"))
    res_status = "✓ Resume loaded — personalised mode" if has_resume else "No resume · upload via Resume Analyzer for personalised advice"
    st.markdown(f'<div class="status-bar"><span class="green-dot"></span>&nbsp; Groq Llama-3.3-70B &nbsp;·&nbsp; {res_status}</div>', unsafe_allow_html=True)

    if has_resume and st.session_state.get("resume_skills"):
        preview = st.session_state.resume_skills[:130].replace("\n", " ")
        st.markdown(f'<div class="resume-banner">✦ Personalised mode active — Skills on file: <strong>{preview}…</strong></div>', unsafe_allow_html=True)

    if not st.session_state.assistant_messages:
        st.markdown('<p class="section-hint">Not sure where to start? Pick a question:</p>', unsafe_allow_html=True)
        for row_i in range(0, len(_STARTER_PROMPTS), 2):
            row = _STARTER_PROMPTS[row_i: row_i + 2]
            c1, c2 = st.columns(2, gap="small")
            for col, chip in zip([c1, c2], row):
                with col:
                    if st.button(chip, key=f"chip_{row_i}_{chip[:18]}", use_container_width=True):
                        st.session_state._pending_msg = chip

    pending = st.session_state.pop("_pending_msg", None)
    if pending:
        st.session_state.assistant_messages.append({"role": "user", "content": pending})
        st.session_state.assistant_input_key += 1
        with st.spinner("CareerForge AI is thinking…"):
            msgs   = _inject_resume_context(st.session_state.assistant_messages)
            system = _build_groq_system_prompt()
            reply  = _call_groq(msgs, system)
        st.session_state.assistant_messages.append({"role": "assistant", "content": reply})
        detected = _extract_roles_from_reply(reply)
        if detected: st.session_state.job_search_role = detected[0]
        st.rerun()

    for msg in st.session_state.assistant_messages:
        if msg["role"] == "user":
            st.markdown('<div class="msg-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-user">{_escape_html(msg["content"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="msg-label-ai">⬡ CareerForge AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="msg-assistant">', unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.assistant_messages:
        st.markdown("<br>", unsafe_allow_html=True)

    col_in, col_send, col_clear = st.columns([6, 1, 1], gap="small")
    with col_in:
        user_input = st.text_input(
            label="msg", label_visibility="collapsed",
            placeholder="Ask about career paths, roles, skills, transitions…",
            key=f"ai_input_{st.session_state.assistant_input_key}",
        )
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.assistant_messages = []
            st.session_state.assistant_input_key += 1
            st.session_state.job_search_role = ""
            st.session_state.job_results = []
            st.rerun()

    if send and user_input and user_input.strip():
        st.session_state._pending_msg = user_input.strip()
        st.rerun()

    if st.session_state.assistant_messages:
        st.markdown("---")
        st.markdown('<p class="section-hint">Quick follow-ups:</p>', unsafe_allow_html=True)
        qa_cols = st.columns(2, gap="small")
        for i, qa in enumerate(_QUICK_FOLLOWUPS):
            with qa_cols[i % 2]:
                if st.button(qa, key=f"qa_{i}", use_container_width=True):
                    st.session_state._pending_msg = qa
                    st.rerun()

    if st.session_state.assistant_messages and st.session_state.get("job_search_role"):
        st.markdown("---")
        st.markdown('<div class="screen-header" style="font-size:1.2rem;">🏢 Live Job Listings</div>', unsafe_allow_html=True)
        jcol1, jcol2, jcol3 = st.columns([3, 2, 1], gap="small")
        with jcol1:
            job_role_input = st.text_input("Role", value=st.session_state.job_search_role,
                                           key="job_role_field", label_visibility="collapsed")
        with jcol2:
            job_location = st.text_input("Location", value="india", key="job_location_field", label_visibility="collapsed")
        with jcol3:
            fetch_jobs = st.button("Search", type="primary", use_container_width=True, key="fetch_jobs_btn")

        if fetch_jobs or (st.session_state.get("job_search_role") and not st.session_state.get("job_results")):
            role_to_search = job_role_input or st.session_state.job_search_role
            with st.spinner(f"Fetching live jobs for '{role_to_search}'…"):
                jobs = fetch_adzuna_jobs(role_to_search, job_location or "india", results=7)
            st.session_state.job_results     = jobs
            st.session_state.job_search_role = role_to_search

        if st.session_state.get("job_results"):
            render_job_cards(st.session_state.job_results, st.session_state.job_search_role,
                             st.session_state.get("job_search_location", "india"),
                             resume_skills=st.session_state.resume_skills)


# =========================================================
# SCREEN: JOB LISTINGS
# =========================================================
_EXP_LEVELS = {
    "fresher":  ("🌱 Fresher / Student",    "0 years — fresh graduate or student"),
    "junior":   ("🔰 Junior (0–2 yrs)",      "Up to 2 years of work experience"),
    "mid":      ("⚡ Mid-Level (3–5 yrs)",   "3–5 years of professional experience"),
    "senior":   ("🚀 Senior (5+ yrs)",       "5 or more years, team lead or specialist"),
}

def _render_experience_onboarding() -> bool:
    st.markdown("""
    <div style="background:#080f18; border:1px solid #1a2535; border-left:4px solid #22D3EE;
                border-radius:14px; padding:28px 28px 20px 28px; margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800; color:#F3F4F6; margin-bottom:8px;">
            👋 Tell us about your experience
        </div>
        <div style="font-size:0.85rem; color:#9CA3AF; line-height:1.6;">
            We'll personalise job listings and show skill-match scores for each listing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2 = st.columns(2, gap="medium")
    with ec1:
        exp_choice = st.radio(
            "Your experience level", options=list(_EXP_LEVELS.keys()),
            format_func=lambda k: _EXP_LEVELS[k][0],
            key="exp_radio_choice", index=0,
        )
        st.caption(_EXP_LEVELS[exp_choice][1])
    with ec2:
        exp_years = st.number_input("Years of work experience", min_value=0, max_value=40, value=0, step=1, key="exp_years_input")
        current_role = st.text_input("Your current / last role (optional)", placeholder="e.g. CS Student, Junior Developer…", key="exp_current_role")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Save Profile & Find Jobs →", type="primary", key="save_exp_profile"):
        st.session_state.user_experience_level = exp_choice
        st.session_state.user_experience_years = int(exp_years)
        st.session_state.exp_profile_done      = True
        st.session_state.job_current_page      = 1
        st.session_state.job_results           = []
        if current_role.strip(): st.session_state["user_current_role"] = current_role.strip()
        st.rerun()
        return True
    return False


def render_jobs_app():
    back_button()
    st.markdown('<div class="screen-header">Live Job Listings</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-subheader">Real jobs from India · Skill match score · Applied history · Adzuna</div>', unsafe_allow_html=True)

    if not st.session_state.get("exp_profile_done"):
        _render_experience_onboarding()
        return

    exp_level = st.session_state.get("user_experience_level", "fresher")
    exp_years = st.session_state.get("user_experience_years", 0)
    exp_label = _EXP_LEVELS.get(exp_level, ("",""))[0]
    applied_count = len(st.session_state.get("applied_jobs", []))

    col_prof, col_edit = st.columns([5, 1], gap="small")
    with col_prof:
        resume_note = " · ⚡ Skill match scores enabled" if st.session_state.resume_skills else ""
        st.markdown(
            f'<div style="background:#0e2a35; border-radius:8px; padding:8px 14px; font-size:0.82rem; color:#22D3EE;">'
            f'🎯 <strong>Your Profile:</strong> {exp_label} · {exp_years} yr{"s" if exp_years != 1 else ""} experience'
            f'{"&nbsp;·&nbsp; ✓ " + str(applied_count) + " applied" if applied_count else ""}'
            f'{resume_note}</div>',
            unsafe_allow_html=True,
        )
    with col_edit:
        if st.button("✏️ Edit", key="edit_exp_profile", use_container_width=True):
            st.session_state.exp_profile_done = False
            st.session_state.job_results = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns([3, 2, 1, 1], gap="small")
    with sc1:
        default_role = st.session_state.get("job_search_role", "")
        if not default_role and st.session_state.get("resume_skills"):
            try:
                top = suggest_top_3_roles(st.session_state.resume_skills, top_k=1)
                if top: default_role = top[0][0]
            except Exception: pass
        search_role = st.text_input("Role", value=default_role,
                                     placeholder="e.g. Data Analyst, Python Developer…",
                                     key="jobs_role_input", label_visibility="collapsed")
    with sc2:
        search_location = st.text_input("Location", value=st.session_state.get("job_search_location","india"),
                                         placeholder="Bangalore, Hyderabad, india…",
                                         key="jobs_location_input", label_visibility="collapsed")
    with sc3:
        results_count = st.selectbox("Results", [5,10,15], key="jobs_count", label_visibility="collapsed")
    with sc4:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True, key="main_search_btn")

    fresher_chips = {
        "🎓 Internships": ["Data Analyst Intern","Software Developer Intern","ML Intern",
                           "Data Science Intern","Web Developer Intern","Python Intern"],
        "💼 Full-Time (Fresher)": ["Software Engineer Fresher","Data Analyst Fresher",
                                    "Python Developer Fresher","Full Stack Developer Fresher",
                                    "ML Engineer Fresher","DevOps Fresher"],
    }
    quick_roles_map = {
        "junior":  ["Junior Data Analyst","Junior Python Developer","Associate DevOps",
                    "Junior ML Engineer","Junior Full Stack","Junior Cloud Engineer"],
        "mid":     ["Data Analyst","Python Developer","DevOps Engineer","ML Engineer",
                    "Full Stack Developer","Cloud Engineer","Data Scientist","Backend Developer"],
        "senior":  ["Senior Data Analyst","Senior ML Engineer","Lead DevOps","Senior Full Stack",
                    "Principal Engineer","Data Engineering Lead","Senior Cloud Architect","Tech Lead"],
    }

    if exp_level == "fresher":
        for section_label, roles in fresher_chips.items():
            st.markdown(f'<p class="section-hint">{section_label}:</p>', unsafe_allow_html=True)
            qr_cols = st.columns(3, gap="small")
            for i, qr in enumerate(roles):
                with qr_cols[i % 3]:
                    if st.button(qr, key=f"qrole_{section_label[:3]}_{i}", use_container_width=True):
                        st.session_state["_job_quick_role"] = qr
    else:
        quick_roles = quick_roles_map.get(exp_level, quick_roles_map["mid"])
        st.markdown('<p class="section-hint">Quick searches for your level:</p>', unsafe_allow_html=True)
        qr_cols = st.columns(4, gap="small")
        for i, qr in enumerate(quick_roles):
            with qr_cols[i % 4]:
                if st.button(qr, key=f"qrole_{i}", use_container_width=True):
                    st.session_state["_job_quick_role"] = qr

    if "_job_quick_role" in st.session_state:
        quick = st.session_state.pop("_job_quick_role")
        st.session_state.job_search_role    = quick
        st.session_state.job_search_location = "india"
        st.session_state.job_current_page   = 1
        st.session_state.job_results        = []
        with st.spinner(f"Finding {exp_label} jobs for '{quick}'…"):
            jobs = fetch_adzuna_jobs(quick, "india", results=int(results_count),
                                     page=1, exp_level=exp_level,
                                     exclude_urls=st.session_state.get("applied_job_urls",[]))
        st.session_state.job_results = jobs
        st.rerun()

    if search_clicked:
        if not search_role.strip():
            st.warning("Please enter a job title to search.")
        else:
            st.session_state.job_search_role     = search_role.strip()
            st.session_state.job_search_location = search_location.strip()
            st.session_state.job_current_page    = 1
            st.session_state.job_results         = []
            with st.spinner(f"Finding {exp_label} jobs for '{search_role}'…"):
                jobs = fetch_adzuna_jobs(search_role.strip(), search_location.strip(),
                                         results=int(results_count), page=1,
                                         exp_level=exp_level,
                                         exclude_urls=st.session_state.get("applied_job_urls",[]))
            st.session_state.job_results = jobs
            st.rerun()

    if not st.session_state.get("job_results") and st.session_state.get("job_search_role"):
        role_r = st.session_state.job_search_role
        loc_r  = st.session_state.get("job_search_location","india")
        page_r = st.session_state.get("job_current_page",1)
        with st.spinner(f"Loading {exp_label} jobs for '{role_r}'…"):
            jobs = fetch_adzuna_jobs(role_r, loc_r, results=int(results_count),
                                     page=page_r, exp_level=exp_level,
                                     exclude_urls=st.session_state.get("applied_job_urls",[]))
        st.session_state.job_results = jobs

    if st.session_state.get("job_results"):
        st.markdown("---")
        role_shown = st.session_state.get("job_search_role","")
        loc_shown  = st.session_state.get("job_search_location","india")
        page_shown = st.session_state.get("job_current_page",1)
        valid_jobs = [j for j in st.session_state.job_results if "error" not in j]

        col_h1, col_h2 = st.columns([4, 1], gap="small")
        with col_h1:
            has_match = bool(st.session_state.resume_skills)
            match_note = " · ⚡ Match scores shown" if has_match else ""
            st.markdown(f'<p class="section-hint">{len(valid_jobs)} <strong>{exp_label}</strong> jobs · '
                        f'<strong>{role_shown}</strong> · {loc_shown.title()} · Page {page_shown}{match_note}</p>',
                        unsafe_allow_html=True)

        render_job_cards(st.session_state.job_results, role_shown, loc_shown,
                         show_applied_btn=True, resume_skills=st.session_state.resume_skills)

        lm_col, _ = st.columns([2, 3])
        with lm_col:
            if st.button("⟳ Load More Jobs", use_container_width=True, key="load_more_jobs"):
                st.session_state.job_current_page += 1
                st.session_state.job_results = []
                st.rerun()
        st.caption(f"Adzuna Jobs API · Page {page_shown} · {exp_label} · Applied jobs excluded")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #4B5563;">
            <div style="font-size:3rem; margin-bottom:16px;">💼</div>
            <div style="font-size:1rem; font-weight:600; color:#9CA3AF;">Search for jobs above</div>
            <div style="font-size:0.82rem; margin-top:8px;">Enter a role title or pick a quick search chip</div>
        </div>
        """, unsafe_allow_html=True)

    applied_jobs = st.session_state.get("applied_jobs", [])
    if applied_jobs:
        st.markdown("---")
        st.markdown(f"### 📋 Applied Jobs History ({len(applied_jobs)} total)")

        avg_match = round(np.mean([j.get("match_pct", 0) for j in applied_jobs]), 1)
        roles_searched = list({j.get("role_searched","") for j in applied_jobs if j.get("role_searched")})
        stat1, stat2, stat3 = st.columns(3)
        with stat1: st.metric("Total Applied", len(applied_jobs))
        with stat2: st.metric("Avg Match Score", f"{avg_match}%")
        with stat3: st.metric("Roles Searched", len(roles_searched))

        st.markdown("#### Applications Log")
        for idx, app in enumerate(reversed(applied_jobs), 1):
            match_pct = app.get("match_pct", 0)
            cls = "match-high" if match_pct >= 60 else ("match-mid" if match_pct >= 35 else "match-low")
            st.markdown(f"""
            <div class="job-card" style="padding:12px 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="job-title" style="font-size:0.88rem;">{idx}. {app.get('title','Unknown')}</span>
                        <span class="match-pill {cls}">⚡ {match_pct}%</span>
                    </div>
                    <div style="font-size:0.72rem; color:#4B5563;">{app.get('date','')}</div>
                </div>
                <div class="job-meta" style="margin-top:4px;">🏢 {app.get('company','—')} &nbsp;·&nbsp; 🔍 searched: {app.get('role_searched','—')}</div>
                <a href="{app.get('url','#')}" target="_blank" style="font-size:0.72rem; color:#22D3EE;">View listing →</a>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear Applied History", key="clear_applied"):
            st.session_state.applied_jobs     = []
            st.session_state.applied_job_urls = []
            st.session_state.job_current_page = 1
            st.session_state.job_results      = []
            st.rerun()

    with st.expander("ℹ️ About Adzuna Jobs API"):
        st.markdown("Adzuna aggregates Indian job listings from Naukri, LinkedIn, Indeed and more. Free tier: 100 calls/day.")


# =========================================================
# SCREEN: ABOUT
# =========================================================
def render_about_app():
    back_button()
    st.markdown('<div class="screen-header">About / Help</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Smart CareerFordge** is an AI-powered platform for job seekers, career changers, and students.

**Platform Modules**

1. **Resume Analyzer** — PDF upload, skill extraction, Resume Score Card (0–100 with letter grade A/B/C/D)
2. **Career Roles Explorer** — Browse {len(role_names)} roles from Career_Path.pdf with roadmaps
3. **Skill Match & Gap** — 4-method cascade matching (exact / alias / token overlap / semantic)
4. **Top Role Suggestions** — AI-ranked role recommendations from your resume
5. **Career Transitions** — LLM-powered gap analysis with learning roadmap
6. **ATS Analysis** — Simulate applicant tracking scoring against a JD
7. **Readiness Dashboard** — 6-dimension career readiness radar chart
8. **Career Assistant** — Groq Llama-3.3-70B chatbot with live job integration
9. **Job Listings** — Adzuna live jobs with **% skill match score** per listing
10. **Job Market Insights** — Live Adzuna job counts per role with bar chart
11. **Salary Trends** — Bar chart comparing avg salary across top 5 recommended roles
12. **AI Cover Letter** — One-click Groq-powered cover letter using resume + JD
13. **Salary Estimator** — Groq estimates salary range by skills + location + experience
14. **LinkedIn Bio Writer** — AI rewrites resume into a professional LinkedIn About section
15. **Feedback & Community** — Star ratings, comments, tag-based reviews, like system & community wall

**Models & APIs**
- Career Assistant + Cover Letter + Salary + LinkedIn: Groq Llama-3.3-70B (free tier)
- Job Listings + Market Insights: Adzuna Jobs API (free tier, 100 calls/day)
- Skill matching: sentence-transformers/all-MiniLM-L6-v2 + ChromaDB
- LLM (roadmap/ATS/transitions/summaries): Groq Llama-3.3-70B (fast, no timeouts)

**Skill Dictionaries** — Built dynamically from Career_Path.pdf at startup.
Currently **{len(STRONG_TECH_KEYWORDS)}** unique skills indexed across **{len(role_names)}** roles.

**Environment Variables needed in .env:**
- `GROQ_API_KEY` — Groq API key for Llama-3.3-70B (used for ALL LLM calls)
- `ADZUNA_APP_ID` — Adzuna application ID for job listings
- `ADZUNA_APP_KEY` — Adzuna application key for job listings
- `HUGGINGFACEHUB_API_TOKEN` — Still needed for HuggingFace Embeddings (sentence-transformers)
- `LANGCHAIN_API_KEY` (optional) — LangSmith tracing
- `LANGCHAIN_TRACING_V2` (optional) — Set to 'true' to enable tracing
- `LANGCHAIN_PROJECT` (optional) — LangSmith project name
- `NTFY_TOPIC` (optional) — Push alert notifications
""")


# =========================================================
# ROUTER
# =========================================================
screen = st.session_state.current_screen
if   screen == "splash":            render_splash()
elif screen == "home":              render_home()
elif screen == "resume_app":        render_resume_app()
elif screen == "roles_app":         render_roles_app()
elif screen == "skill_match_app":   render_skill_match_app()
elif screen == "top_roles_app":     render_top_roles_app()
elif screen == "transition_app":    render_transition_app()
elif screen == "ats_app":           render_ats_app()
elif screen == "readiness_app":     render_readiness_app()
elif screen == "assistant_app":     render_assistant_app()
elif screen == "jobs_app":          render_jobs_app()
elif screen == "market_app":        render_market_app()
elif screen == "salary_chart_app":  render_salary_chart_app()
elif screen == "coverletter_app":   render_coverletter_app()
elif screen == "salary_est_app":    render_salary_est_app()
elif screen == "linkedin_app":      render_linkedin_app()
elif screen == "feedback_app":      render_feedback_app()
elif screen == "about_app":         render_about_app()
else:                               render_splash()
