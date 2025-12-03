#app.py
# ---------------------------
# Simple Streamlit Dashboard
# ---------------------------
# Run with:
#   streamlit run app.py
#
# Then open the URL Streamlit prints in your terminal.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="My Simple Dashboard",
    page_icon="📊",
    layout="wide",
)


# ---------------------------
# DATA GENERATION / LOADING
# ---------------------------
@st.cache_data
def load_data(num_days: int = 60) -> pd.DataFrame:
    """
    Creates some example time series data.
    Replace this function with your own data loading logic later.
    """
    np.random.seed(42)
    today = datetime.today().date()
    dates = [today - timedelta(days=i) for i in range(num_days)][::-1]

    categories = ["Product A", "Product B", "Product C"]
    rows = []

    for d in dates:
        for cat in categories:
            value = np.random.normal(loc=100, scale=20)
            value = max(value, 0)  # avoid negative values
            rows.append(
                {
                    "date": d,
                    "category": cat,
                    "value": round(value, 2),
                    "volume": np.random.randint(10, 200),
                }
            )

    df = pd.DataFrame(rows)
    # Fix: Convert the 'date' column to pandas datetime objects for consistent comparison
    df['date'] = pd.to_datetime(df['date'])
    return df


data = load_data()


# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("Controls ⚙️")

# Date filter
min_date = data["date"].min()
max_date = data["date"].max()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Category filter
all_categories = sorted(data["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Categories",
    options=all_categories,
    default=all_categories,
)

# Metric selector (for charts)
metric = st.sidebar.selectbox("Metric to visualize", ["value", "volume"])

# Sidebar info
with st.sidebar.expander("About this app"):
    st.write(
        """
        This is a simple Streamlit dashboard template.

        **How to customize:**
        - Replace the fake data in `load_data()`.
        - Add or remove charts in the main area.
        - Adjust sidebar filters and controls.
        """
    )


# ---------------------------
# FILTER DATA
# ---------------------------
filtered_data = data.copy()

# Apply date filter
if isinstance(start_date, tuple):
    # Streamlit can return (start, end) for date_input in some versions
    start_date, end_date = start_date

filtered_data = filtered_data[
    (filtered_data["date"] >= pd.to_datetime(start_date))
    & (filtered_data["date"] <= pd.to_datetime(end_date))
]

# Apply category filter
if selected_categories:
    filtered_data = filtered_data[filtered_data["category"].isin(selected_categories)]


# ---------------------------
# HEADER
# ---------------------------
st.title("📊 My Simple Dashboard")
st.caption("Use this as a starting point and customize it for your own data.")


# ---------------------------
# TOP-LEVEL METRICS
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    total_metric = filtered_data[metric].sum()
    st.metric(
        label=f"Total {metric.capitalize()}",
        value=f"{total_metric:,.0f}",
    )

with col2:
    avg_metric = filtered_data[metric].mean()
    st.metric(
        label=f"Average {metric.capitalize()}",
        value=f"{avg_metric:,.1f}",
    )

with col3:
    # Last day vs previous day comparison (if enough data)
    latest_date = filtered_data["date"].max()
    prev_date = latest_date - timedelta(days=1)

    latest_val = filtered_data.loc[
        filtered_data["date"] == latest_date, metric
    ].sum()

    prev_val = filtered_data.loc[
        filtered_data["date"] == prev_date, metric
    ].sum()

    delta = latest_val - prev_val if not np.isnan(prev_val) else 0
    st.metric(
        label=f"{metric.capitalize()} (last day)",
        value=f"{latest_val:,.0f}",
        delta=f"{delta:,.0f}",
    )


# ---------------------------
# MAIN TABS
# ---------------------------
tab_overview, tab_details, tab_raw = st.tabs(
    ["📈 Overview", "📊 Details", "📄 Raw Data"]
)


# ---------------------------
# OVERVIEW TAB
# ---------------------------
with tab_overview:
    st.subheader("Trend over Time")

    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Time series chart – simple line chart by date
        # You can customize or replace this with Altair/Plotly if you want.
        ts = (
            filtered_data.groupby("date")[metric]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        ts = ts.set_index("date")

        st.line_chart(ts, height=350)

        st.markdown("---")

        st.subheader("By Category")

        # Aggregate by category for a bar chart
        by_cat = filtered_data.groupby("category")[metric].sum().reset_index()
        by_cat = by_cat.set_index("category")

        st.bar_chart(by_cat, height=350)


# ---------------------------
# DETAILS TAB
# ---------------------------
with tab_details:
    st.subheader("Aggregated View")

    agg_type = st.radio(
        "Aggregation level",
        ["Daily by Category", "Overall by Category"],
        horizontal=True,
    )

    if agg_type == "Daily by Category":
        grouped = (
            filtered_data.groupby(["date", "category"])[metric]
            .sum()
            .reset_index()
            .sort_values(["date", "category"])
        )
        st.dataframe(grouped, use_container_width=True, height=350)
    else:
        grouped = (
            filtered_data.groupby("category")[metric]
            .agg(["sum", "mean", "min", "max"])
            .reset_index()
        )
        grouped.columns = [
            "Category",
            f"Total {metric}",
            f"Avg {metric}",
            f"Min {metric}",
            f"Max {metric}",
        ]
        st.dataframe(grouped, use_container_width=True, height=350)

    st.markdown("---")
    st.subheader("Download Data")

    csv = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )


# ---------------------------
# RAW DATA TAB
# ---------------------------
with tab_raw:
    st.subheader("Raw Data (Unfiltered & Filtered)")

    st.markdown("**Filtered data (based on sidebar controls):**")
    st.dataframe(filtered_data, use_container_width=True, height=350)

    with st.expander("Show full raw dataset (ignores filters)"):
        st.dataframe(data, use_container_width=True, height=350)


# ---------------------------
# FOOTER / NOTES
# ---------------------------
st.markdown("---")
st.caption(
    "Template built with ❤️ using Streamlit. "
    "Edit `app1.py` to connect to your own data and customize the layout."
)