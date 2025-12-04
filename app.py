# app.py
# Event Study Dashboard for Senator Trades

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Senator Trades Event Study",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Senator Trades Event Study Dashboard")
st.caption(
    "Analyze stock performance around senators' trades and compare to the S&P 500 and Nasdaq."
)

# ---------------------------
# DATA LOADING
# ---------------------------

@st.cache_data
def load_event_data(csv_path: str = "results_df.csv") -> pd.DataFrame:
    """Load your precomputed event-study data from CSV."""
    df = pd.read_csv(csv_path)

    # Clean types
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

    # Make event_id an integer if possible
    if "event_id" in df.columns:
        try:
            df["event_id"] = df["event_id"].astype(int)
        except Exception:
            pass

    # Fill missing party if any
    if "Party" in df.columns:
        df["Party"] = df["Party"].fillna("Unknown")

    return df


@st.cache_data
def get_index_returns(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download daily returns for S&P 500 (^GSPC) and Nasdaq (^IXIC)
    between start_date and end_date using yfinance.
    """
    tickers = ["^GSPC", "^IXIC"]
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )

    # yfinance returns a DataFrame with columns like ('Adj Close', '^GSPC')
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"].copy()
    else:
        # Single index case; assume these are adjusted close prices
        adj_close = data.copy()

    # Normalize column names
    col_map = {}
    for col in adj_close.columns:
        if "GSPC" in col:
            col_map[col] = "SP500"
        elif "IXIC" in col:
            col_map[col] = "NASDAQ"
        else:
            col_map[col] = col
    adj_close = adj_close.rename(columns=col_map)

    # Compute daily returns
    ret = adj_close.pct_change().reset_index()
    ret.rename(columns={"Date": "Date"}, inplace=True)
    ret["Date"] = ret["Date"].dt.date

    # Keep only what we need
    expected_cols = []
    if "SP500" in ret.columns:
        expected_cols.append("SP500")
    if "NASDAQ" in ret.columns:
        expected_cols.append("NASDAQ")

    return ret[["Date"] + expected_cols]


@st.cache_data
def compute_senator_overall_stats(
    df: pd.DataFrame,
    idx_ret: pd.DataFrame,
    senator: str,
    start_offset: int,
    end_offset: int,
) -> pd.DataFrame:
    """
    For a given senator and event window, compute cumulative event-window returns
    for each event, and summarize averages.
    """
    df_sen = df[df["senator"] == senator].copy()

    # Merge index returns for all relevant dates once
    merged = df_sen.merge(idx_ret, on="Date", how="left")

    # Ensure offsets are within full range
    merged = merged[
        (merged["Offset"] >= start_offset) & (merged["Offset"] <= end_offset)
    ]

    # Group by event
    event_groups = merged.groupby(["event_id", "ticker", "event_date"], as_index=False)

    summary_rows = []
    for (event_id, ticker, event_date), g in event_groups:
        g_sorted = g.sort_values("Date")

        # Stock cumulative return in the window
        stock_cum = (1 + g_sorted["Ret"].fillna(0)).prod() - 1

        # SP500
        if "SP500" in g_sorted.columns:
            sp500_cum = (1 + g_sorted["SP500"].fillna(0)).prod() - 1
        else:
            sp500_cum = np.nan

        # NASDAQ
        if "NASDAQ" in g_sorted.columns:
            nasdaq_cum = (1 + g_sorted["NASDAQ"].fillna(0)).prod() - 1
        else:
            nasdaq_cum = np.nan

        summary_rows.append(
            {
                "event_id": event_id,
                "ticker": ticker,
                "event_date": event_date,
                "stock_cum_return": stock_cum,
                "sp500_cum_return": sp500_cum,
                "nasdaq_cum_return": nasdaq_cum,
                "excess_vs_sp500": stock_cum - sp500_cum
                if pd.notna(sp500_cum)
                else np.nan,
                "excess_vs_nasdaq": stock_cum - nasdaq_cum
                if pd.notna(nasdaq_cum)
                else np.nan,
            }
        )

    if not summary_rows:
        return pd.DataFrame()

    return pd.DataFrame(summary_rows)


# ---------------------------
# LOAD DATA
# ---------------------------

try:
    data = load_event_data("results_df.csv")  # change path if needed
except FileNotFoundError:
    st.error(
        "Could not find `results_df.csv`. "
        "Place it in the same folder as `app.py` or adjust the path in `load_event_data()`."
    )
    st.stop()

min_date = data["Date"].min()
max_date = data["Date"].max()

# Pre-load index returns for the full date range of your study
with st.spinner("Fetching S&P 500 and Nasdaq data..."):
    index_returns = get_index_returns(min_date, max_date)

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------

st.sidebar.header("Filters")

# Party filter
parties = sorted(data["Party"].dropna().unique().tolist())
selected_party = st.sidebar.multiselect(
    "Party",
    options=parties,
    default=parties,  # show all by default
)

df_party = data[data["Party"].isin(selected_party)] if selected_party else data.copy()

# Senator selection
senators = sorted(df_party["senator"].dropna().unique().tolist())

if not senators:
    st.warning("No senators available with the current party filter.")
    st.stop()

selected_senator = st.sidebar.selectbox("Senator", options=senators)

df_senator = df_party[df_party["senator"] == selected_senator].copy()

# Event window selection (in days relative to event_date)
min_offset = int(df_senator["Offset"].min())
max_offset = int(df_senator["Offset"].max())

start_offset, end_offset = st.sidebar.slider(
    "Event window (days from event date)",
    min_value=min_offset,
    max_value=max_offset,
    value=(-5, 5),
)

# Event selection: unique (event_id, ticker, event_date) combos
event_list = (
    df_senator[["event_id", "ticker", "event_date"]]
    .drop_duplicates()
    .sort_values(["event_date", "ticker", "event_id"])
)

if event_list.empty:
    st.warning("No events available for this senator.")
    st.stop()

event_labels = [
    f"Event {int(row.event_id)} | {row.ticker} | {row.event_date}"
    for _, row in event_list.iterrows()
]
event_id_mapping = {
    label: int(row.event_id) for label, (_, row) in zip(event_labels, event_list.iterrows())
}

selected_event_label = st.sidebar.selectbox("Select trade/event", options=event_labels)
selected_event_id = event_id_mapping[selected_event_label]

# ---------------------------
# SELECTED EVENT DATA
# ---------------------------

event_df = df_senator[df_senator["event_id"] == selected_event_id].copy()
event_df = event_df.sort_values("Date")

# Merge with index returns
event_merged = event_df.merge(index_returns, on="Date", how="left")

# Apply event window offsets
event_window = event_merged[
    (event_merged["Offset"] >= start_offset) & (event_merged["Offset"] <= end_offset)
].copy()

if event_window.empty:
    st.warning("No data in the chosen event window for this event.")
    st.stop()

# Identify ticker and event date for display
event_ticker = event_window["ticker"].iloc[0]
event_date = event_window["event_date"].iloc[0]
party = event_window["Party"].iloc[0]


# ---------------------------
# TOP SUMMARY / OVERALL STATS
# ---------------------------

st.subheader("Summary for Selected Senator")

overall_stats = compute_senator_overall_stats(
    data, index_returns, selected_senator, start_offset, end_offset
)

col1, col2, col3, col4, col5 = st.columns(5)

if not overall_stats.empty:
    avg_stock = overall_stats["stock_cum_return"].mean()
    avg_sp500 = overall_stats["sp500_cum_return"].mean()
    avg_nasdaq = overall_stats["nasdaq_cum_return"].mean()
    avg_excess_sp = overall_stats["excess_vs_sp500"].mean()
    avg_excess_nd = overall_stats["excess_vs_nasdaq"].mean()

    col1.metric(
        "Avg Stock Return (Event Window)",
        f"{avg_stock * 100:,.2f}%",
    )
    col2.metric(
        "Avg S&P 500 Return (Event Window)",
        f"{avg_sp500 * 100:,.2f}%",
    )
    col3.metric(
        "Avg Nasdaq Return (Event Window)",
        f"{avg_nasdaq * 100:,.2f}%",
    )
    col4.metric(
        "Avg Excess vs S&P 500",
        f"{avg_excess_sp * 100:,.2f}%",
    )
    col5.metric(
        "Avg Excess vs Nasdaq",
        f"{avg_excess_nd * 100:,.2f}%",
    )
else:
    col1.info("Not enough data to compute senator-level stats for this window.")


st.markdown("---")

# ---------------------------
# SELECTED EVENT DETAILS
# ---------------------------

st.subheader("Selected Event Details")

st.write(
    f"**Senator:** {selected_senator}  "
    f" | **Party:** {party}  "
    f" | **Ticker:** `{event_ticker}`  "
    f" | **Event Date:** {event_date}  "
    f" | **Event ID:** {selected_event_id}  "
)

# Compute cumulative returns for the event window
event_window = event_window.sort_values("Offset")

event_window["stock_cum_ret"] = (1 + event_window["Ret"].fillna(0)).cumprod() - 1

if "SP500" in event_window.columns:
    event_window["sp500_cum_ret"] = (
        (1 + event_window["SP500"].fillna(0)).cumprod() - 1
    )
else:
    event_window["sp500_cum_ret"] = np.nan

if "NASDAQ" in event_window.columns:
    event_window["nasdaq_cum_ret"] = (
        (1 + event_window["NASDAQ"].fillna(0)).cumprod() - 1
    )
else:
    event_window["nasdaq_cum_ret"] = np.nan

# ---------------------------
# EVENT-LEVEL METRICS
# ---------------------------

final_row = event_window.iloc[-1]

ec1, ec2, ec3 = st.columns(3)
ec1.metric(
    "Stock Cumulative Return (Selected Event)",
    f"{final_row['stock_cum_ret'] * 100:,.2f}%",
)
ec2.metric(
    "S&P 500 Cumulative Return (Same Window)",
    f"{final_row['sp500_cum_ret'] * 100:,.2f}%",
)
ec3.metric(
    "Nasdaq Cumulative Return (Same Window)",
    f"{final_row['nasdaq_cum_ret'] * 100:,.2f}%",
)


# ---------------------------
# PLOTS
# ---------------------------

tab1, tab2, tab3 = st.tabs(
    ["📈 Stock vs Indexes", "📊 AR & CAR", "📄 Event Data Table"]
)

with tab1:
    st.markdown("### Cumulative Returns Around the Event")

    plot_df = event_window[["Offset", "stock_cum_ret", "sp500_cum_ret", "nasdaq_cum_ret"]].copy()
    plot_df = plot_df.set_index("Offset")

    st.line_chart(plot_df)

with tab2:
    st.markdown("### Abnormal Returns (AR) and Cumulative Abnormal Returns (CAR)")

    if "ar" in event_window.columns and "car" in event_window.columns:
        ar_car_df = event_window[["Offset", "ar", "car"]].copy()
        ar_car_df = ar_car_df.set_index("Offset")
        st.line_chart(ar_car_df)
    else:
        st.info("AR and CAR columns not found in the data.")

with tab3:
    st.markdown("### Underlying Event Data")
    st.dataframe(
        event_window[
            [
                "Date",
                "Offset",
                "Ret",
                "SP500",
                "NASDAQ",
                "stock_cum_ret",
                "sp500_cum_ret",
                "nasdaq_cum_ret",
                "ar",
                "car",
                "sar",
                "scar",
            ]
        ],
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "Tip: adjust the event window in the sidebar to see how results change. "
    "You can also change party and senator to explore different trading patterns."
)
