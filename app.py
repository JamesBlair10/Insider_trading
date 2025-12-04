import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Senator Trades Event Study Dashboard",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Senator Trades Event Study Dashboard")
st.caption(
    "Analyze stock performance around senators' trades using event-study data."
)

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results_df.csv"   # adjust if you put it in a subfolder

@st.cache_data
def load_event_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

    # Clean event_id and Offset
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(int)
    if "Offset" in df.columns:
        df["Offset"] = df["Offset"].astype(int)

    # Ensure Party + senator columns exist
    if "Party" in df.columns:
        df["Party"] = df["Party"].fillna("Unknown")
    else:
        df["Party"] = "Unknown"

    if "senator" not in df.columns:
        df["senator"] = "Unknown"

    return df


if not CSV_PATH.exists():
    st.error(
        f"Could not find results_df.csv at {CSV_PATH}. "
        "Make sure it is committed to the GitHub repo in the same folder as app.py "
        "or adjust CSV_PATH in the code."
    )
    st.stop()

data = load_event_data(str(CSV_PATH))

# -------------------------------------------------
# HELPER: OVERALL SENATOR STATS
# -------------------------------------------------
@st.cache_data
def compute_senator_overall_stats(
    df: pd.DataFrame, senator: str, start_offset: int, end_offset: int
) -> pd.DataFrame:
    """
    For a given senator and event window, compute cumulative event-window stock
    returns and CAR/SCAR for each event.
    """
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
    for (event_id, ticker, event_date), g in grouped:
        g_sorted = g.sort_values("Offset")

        # Cumulative raw stock return over window
        stock_cum_return = (1 + g_sorted["Ret"].fillna(0)).prod() - 1

        # CAR / SCAR at end of window (last row)
        last_row = g_sorted.iloc[-1]
        car_end = last_row.get("car", np.nan)
        scar_end = last_row.get("scar", np.nan)

        rows.append(
            dict(
                event_id=event_id,
                ticker=ticker,
                event_date=event_date,
                stock_cum_return=stock_cum_return,
                car_end=car_end,
                scar_end=scar_end,
            )
        )

    return pd.DataFrame(rows)


# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Filters")

# Party filter
parties = sorted(data["Party"].dropna().unique().tolist())
selected_parties = st.sidebar.multiselect(
    "Party",
    options=parties,
    default=parties,
)

df_party = data[data["Party"].isin(selected_parties)] if selected_parties else data

# Senator filter
senators = sorted(df_party["senator"].dropna().unique().tolist())
if not senators:
    st.warning("No senators available with the current party filter.")
    st.stop()

selected_senator = st.sidebar.selectbox("Senator", options=senators)
df_senator = df_party[df_party["senator"] == selected_senator].copy()

# Event window
min_offset = int(df_senator["Offset"].min())
max_offset = int(df_senator["Offset"].max())
default_start = max(min_offset, -5)
default_end = min(max_offset, 5)

start_offset, end_offset = st.sidebar.slider(
    "Event window (days relative to event date)",
    min_value=min_offset,
    max_value=max_offset,
    value=(default_start, default_end),
)

# Event selection: (event_id, ticker, event_date)
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

selected_event_label = st.sidebar.selectbox("Select trade / event", options=event_labels)
selected_event_id = event_id_mapping[selected_event_label]


# -------------------------------------------------
# OVERALL SENATOR PERFORMANCE
# -------------------------------------------------
st.subheader("Overall Event-Study Performance for Selected Senator")

overall_stats = compute_senator_overall_stats(
    data, selected_senator, start_offset, end_offset
)

col1, col2, col3, col4 = st.columns(4)

if overall_stats.empty:
    col1.info("Not enough data to compute overall statistics for this window.")
else:
    avg_stock = overall_stats["stock_cum_return"].mean()
    med_stock = overall_stats["stock_cum_return"].median()
    avg_car = overall_stats["car_end"].mean()
    avg_scar = overall_stats["scar_end"].mean()
    n_events = len(overall_stats)

    col1.metric(
        "Avg Stock Cumulative Return (per event)",
        f"{avg_stock*100:,.2f}%",
    )
    col2.metric(
        "Median Stock Cumulative Return (per event)",
        f"{med_stock*100:,.2f}%",
    )
    col3.metric(
        "Avg CAR at End of Window",
        f"{avg_car*100:,.2f}%" if pd.notna(avg_car) else "N/A",
    )
    col4.metric(
        "Number of Events Analyzed",
        f"{n_events}",
    )

    with st.expander("Event-by-event summary"):
        st.dataframe(
            overall_stats.sort_values("event_date"),
            use_container_width=True,
        )

st.markdown("---")

# -------------------------------------------------
# SELECTED EVENT DETAILS
# -------------------------------------------------
st.subheader("Selected Event Details")

event_df = df_senator[df_senator["event_id"] == selected_event_id].copy()
event_df = event_df.sort_values("Offset")

event_window = event_df[
    (event_df["Offset"] >= start_offset) & (event_df["Offset"] <= end_offset)
].copy()

if event_window.empty:
    st.warning("No data in the chosen event window for this event.")
    st.stop()

event_ticker = event_window["ticker"].iloc[0]
event_date = event_window["event_date"].iloc[0]
party = event_window["Party"].iloc[0]

st.write(
    f"**Senator:** {selected_senator}  "
    f"| **Party:** {party}  "
    f"| **Ticker:** `{event_ticker}`  "
    f"| **Event Date:** {event_date}  "
    f"| **Event ID:** {selected_event_id}"
)

# Compute cumulative stock return within the window
event_window = event_window.sort_values("Offset")
event_window["stock_cum_ret"] = (1 + event_window["Ret"].fillna(0)).cumprod() - 1

# Event-level metrics
final_row = event_window.iloc[-1]
event_cum_ret = final_row["stock_cum_ret"]
event_car_end = final_row.get("car", np.nan)
event_scar_end = final_row.get("scar", np.nan)

ec1, ec2, ec3 = st.columns(3)
ec1.metric(
    "Stock Cumulative Return (Selected Event & Window)",
    f"{event_cum_ret*100:,.2f}%",
)
ec2.metric(
    "CAR at End of Window",
    f"{event_car_end*100:,.2f}%" if pd.notna(event_car_end) else "N/A",
)
ec3.metric(
    "SCAR at End of Window",
    f"{event_scar_end:,.2f}" if pd.notna(event_scar_end) else "N/A",
)

# -------------------------------------------------
# PLOTS & DATA
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 Cumulative Return", "📊 AR / CAR / SCAR", "📄 Event Data"]
)

with tab1:
    st.markdown("### Cumulative Stock Return Around the Event")
    plot_df = event_window[["Offset", "stock_cum_ret"]].set_index("Offset")
    st.line_chart(plot_df)

with tab2:
    st.markdown("### Abnormal Returns and Cumulative Abnormal Returns")
    cols_to_plot = []
    if "ar" in event_window.columns:
        cols_to_plot.append("ar")
    if "car" in event_window.columns:
        cols_to_plot.append("car")
    if "sar" in event_window.columns:
        cols_to_plot.append("sar")
    if "scar" in event_window.columns:
        cols_to_plot.append("scar")

    if cols_to_plot:
        ar_car_df = event_window[["Offset"] + cols_to_plot].set_index("Offset")
        st.line_chart(ar_car_df)
    else:
        st.info("No AR/CAR/SAR/SCAR columns found in the data.")

with tab3:
    st.markdown("### Raw Event Data (Selected Window)")
    cols = [
        "Date",
        "Offset",
        "Ret",
        "ar",
        "car",
        "sar",
        "scar",
        "Volume",
        "Mkt-RF",
        "SMB",
        "HML",
        "RF",
        "pred_ret_col",
        "estimation_std",
        "stock_cum_ret",
    ]
    existing_cols = [c for c in cols if c in event_window.columns]
    st.dataframe(
        event_window[existing_cols],
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "Use the sidebar to change party, senator, event, and event window to explore different trades."
)
