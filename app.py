##############################################################
# Senator Trades Event Study Dashboard + Gemini News Summary #
# Using Marketaux for finance news scraping                 #
##############################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Gemini (google-generativeai)
try:
    import google.generativeai as genai
except ImportError:
    genai = None


##############################################################
# PAGE CONFIG
##############################################################

st.set_page_config(
    page_title="Senator Trades Event Study Dashboard",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Senator Trades Event Study Dashboard")
st.caption(
    "Analyze stock performance around senators' trades, view event studies, and generate post-trade news summaries using Gemini."
)


##############################################################
# PATHS & API KEYS
##############################################################

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results_df.csv"  # <-- Ensure this file exists in your repo

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")

if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


##############################################################
# LOAD EVENT DATA
##############################################################

@st.cache_data
def load_event_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(int)

    if "Offset" in df.columns:
        df["Offset"] = df["Offset"].astype(int)

    if "Party" not in df.columns:
        df["Party"] = "Unknown"
    else:
        df["Party"] = df["Party"].fillna("Unknown")

    if "senator" not in df.columns:
        df["senator"] = "Unknown"

    return df


if not CSV_PATH.exists():
    st.error(
        f"Could not find results_df.csv at {CSV_PATH}. "
        "Place it in your GitHub repository next to app.py, then redeploy."
    )
    st.stop()

data = load_event_data(str(CSV_PATH))


##############################################################
# OVERALL SENATOR STATS
##############################################################

@st.cache_data
def compute_senator_overall_stats(df: pd.DataFrame, senator: str,
                                  start_offset: int, end_offset: int) -> pd.DataFrame:

    df_sen = df[df["senator"] == senator].copy()
    if df_sen.empty:
        return pd.DataFrame()

    df_window = df_sen[
        (df_sen["Offset"] >= start_offset) & (df_sen["Offset"] <= end_offset)
    ].copy()
    if df_window.empty:
        return pd.DataFrame()

    grouped = df_window.groupby(["event_id", "ticker", "event_date"], as_index=False)

    rows = []
    for (eid, ticker, event_date), g in grouped:
        g_sorted = g.sort_values("Offset")

        stock_cum_return = (1 + g_sorted["Ret"].fillna(0)).prod() - 1
        car = g_sorted.iloc[-1].get("car", np.nan)
        scar = g_sorted.iloc[-1].get("scar", np.nan)

        rows.append(dict(
            event_id=eid,
            ticker=ticker,
            event_date=event_date,
            stock_cum_return=stock_cum_return,
            car_end=car,
            scar_end=scar,
        ))

    return pd.DataFrame(rows)


##############################################################
# MARKETaux NEWS SCRAPER
##############################################################

@st.cache_data(show_spinner=False)
def fetch_news_articles(ticker: str, start_dt: datetime, end_dt: datetime) -> list:
    """
    Fetch news articles from Marketaux for the given ticker after the trade date.
    """
    if not MARKETAUX_API_KEY:
        return []

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": ticker,
        "published_after": start_dt.isoformat(),
        "published_before": end_dt.isoformat(),
        "language": "en",
        "filter_entities": "true",
        "api_token": MARKETAUX_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("data", [])
    except Exception:
        return []

    simplified = []
    for a in articles:
        simplified.append(
            {
                "title": a.get("headline") or a.get("title"),
                "description": a.get("summary"),
                "source": a.get("source"),
                "url": a.get("url"),
                "publishedAt": a.get("published_at"),
            }
        )
    return simplified


##############################################################
# GEMINI NEWS SUMMARIZER
##############################################################

@st.cache_data(show_spinner=False)
def summarize_news_with_gemini(ticker: str, event_date, articles, days):
    if genai is None or not GEMINI_API_KEY:
        return "Gemini is not configured."

    if not articles:
        return f"No news found for {ticker} in the {days} days following {event_date}."

    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = f"""
A U.S. senator executed a trade in {ticker} on {event_date}.
Below are news articles about {ticker} after the trade date.
Write a concise 4–6 sentence summary addressing:

• Major corporate developments (earnings, guidance, legal changes, etc.)
• Major macro / economic events affecting the stock
• Whether sentiment was positive, negative, or mixed

Articles:
{articles}
"""

    try:
        r = model.generate_content(prompt)
        return r.text.strip()
    except Exception as e:
        return f"Gemini request failed: {e}"


##############################################################
# SIDEBAR FILTERS
##############################################################

st.sidebar.header("Filters")

parties = sorted(data["Party"].unique())
selected_parties = st.sidebar.multiselect("Party", parties, default=parties)

df_filtered = data[data["Party"].isin(selected_parties)] if selected_parties else data

senators = sorted(df_filtered["senator"].unique())
selected_senator = st.sidebar.selectbox("Senator", senators)

df_sen = df_filtered[df_filtered["senator"] == selected_senator]

min_offset = int(df_sen["Offset"].min())
max_offset = int(df_sen["Offset"].max())

start_offset, end_offset = st.sidebar.slider(
    "Event study window (days around trade)",
    min_value=min_offset, max_value=max_offset, value=(-5, 5)
)

news_window_days = st.sidebar.slider(
    "News search window after trade (days)",
    min_value=1, max_value=30, value=7
)

events = df_sen[["event_id", "ticker", "event_date"]].drop_duplicates()
events = events.sort_values(["event_date", "ticker", "event_id"])

event_label = st.sidebar.selectbox(
    "Trade / Event",
    [f"{eid} | {t} | {d}" for eid, t, d in events.values]
)
selected_event_id = int(event_label.split("|")[0].strip())


##############################################################
# OVERALL METRICS
##############################################################

st.subheader("Overall event performance for selected senator")

overall = compute_senator_overall_stats(data, selected_senator, start_offset, end_offset)

col1, col2, col3 = st.columns(3)
if overall.empty:
    col1.info("No data available.")
else:
    col1.metric("Avg cumulative return", f"{overall.stock_cum_return.mean()*100:,.2f}%")
    col2.metric("Median cumulative return", f"{overall.stock_cum_return.median()*100:,.2f}%")
    col3.metric("# Trades analyzed", len(overall))


##############################################################
# SELECTED EVENT DETAILS
##############################################################

event_df = df_sen[df_sen["event_id"] == selected_event_id].sort_values("Offset")
event_window = event_df[
    (event_df["Offset"] >= start_offset) & (event_df["Offset"] <= end_offset)
].copy()

ticker = event_window["ticker"].iloc[0]
event_date = event_window["event_date"].iloc[0]

st.subheader(f"Selected Trade: {ticker} on {event_date}")
event_window["stock_cum_ret"] = (1 + event_window["Ret"]).cumprod() - 1
final = event_window.iloc[-1]

st.metric("Cumulative return", f"{final.stock_cum_ret*100:,.2f}%")


##############################################################
# TABS
##############################################################

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cumulative Return",
    "📊 AR / CAR / SCAR",
    "📄 Event Data",
    "📰 Gemini News Summary"
])

with tab1:
    st.line_chart(event_window.set_index("Offset")["stock_cum_ret"])

with tab2:
    cols = [c for c in ["ar", "car", "sar", "scar"] if c in event_window]
    if cols:
        st.line_chart(event_window.set_index("Offset")[cols])
    else:
        st.info("No AR / CAR data columns available.")

with tab3:
    st.dataframe(event_window, use_container_width=True)

with tab4:
    st.write("Click below to generate a news summary after this trade.")

    if st.button("Generate Gemini summary"):
        with st.spinner("Fetching and summarizing news..."):
            start = datetime.combine(event_date, datetime.min.time()) + timedelta(days=1)
            end = start + timedelta(days=news_window_days)

            articles = fetch_news_articles(ticker, start, end)
            summary = summarize_news_with_gemini(ticker, event_date, articles, news_window_days)

        st.subheader("Gemini summary")
        st.write(summary)

        if articles:
            with st.expander("Raw article data"):
                st.dataframe(pd.DataFrame(articles))


##############################################################
st.caption("Use the sidebar to explore senators, trades, and event windows.")
