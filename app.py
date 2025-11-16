"""
app.py

Usage:
    streamlit run app.py

This Streamlit app:
- Loads cleaned.parquet and aggregation parquet files
- Shows sidebar controls: date range, ticker selection, sector selection
- Renders charts (Altair) for:
    * Time series avg_close by ticker (agg1)
    * Avg volume by sector (agg2)
    * Table and histogram of daily returns (agg3)
"""

import streamlit as st
import polars as pl
import altair as alt
import pandas as pd
from datetime import date

CLEANED_PARQUET = "data/cleaned.parquet"
AGG1 = "data/agg1.parquet"
AGG2 = "data/agg2.parquet"
AGG3 = "data/agg3.parquet"

st.set_page_config(page_title="Stock Aggregates Dashboard", layout="wide")

st.title("Stock Aggregates Dashboard")

# Load data (with checks)
def load_parquet_if_exists(path):
    try:
        return pl.read_parquet(path)
    except Exception:
        return None

agg1 = load_parquet_if_exists(AGG1)
agg2 = load_parquet_if_exists(AGG2)
agg3 = load_parquet_if_exists(AGG3)
cleaned = load_parquet_if_exists(CLEANED_PARQUET)

# Provide options based on available data
available_tickers = []
if agg1 is not None and not agg1.is_empty():
    if "ticker" in [c.lower() for c in agg1.columns]:
        available_tickers = (
            agg1
            .select("ticker")
            .drop_nulls()
            .unique()
            .sort("ticker")
            .to_series()
            .to_list()
        )

# Sidebar filters
st.sidebar.header("Filters")
if cleaned is not None and "date" in [c.lower() for c in cleaned.columns]:
    # find date column in cleaned (exact name)
    date_cols = [c for c in cleaned.columns if "date" in c.lower() or c.lower() in ("dt", "day", "timestamp")]
    if date_cols:
        dcol = date_cols[0]
        min_date = cleaned.select(pl.col(dcol).min()).to_series()[0].to_pydatetime().date()
        max_date = cleaned.select(pl.col(dcol).max()).to_series()[0].to_pydatetime().date()
        date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None
else:
    date_range = None

tickers = st.sidebar.multiselect("Tickers", options=available_tickers, default=available_tickers[:5])
# Main content


st.header("Avg Close Time Series - Aggregation 1")
if agg1 is None or agg1.is_empty():
    st.info("agg1.parquet not found or empty. Run the data pipeline.")
else:
    df1 = agg1.to_pandas()
    # ensure date column recognized
    date_col = [c for c in df1.columns if "date" in c.lower() or c.lower() in ("dt", "day", "timestamp")][0]
    df1[date_col] = pd.to_datetime(df1[date_col])
    if tickers:
        df1 = df1[df1["ticker"].isin(tickers)]
    if date_range:
        start, end = date_range
        df1 = df1[(df1[date_col] >= pd.Timestamp(start)) & (df1[date_col] <= pd.Timestamp(end))]
    if df1.empty:
        st.write("No data for selected filters.")
    else:
        chart = alt.Chart(df1).mark_line(point=True).encode(
            x=alt.X(date_col, title="Date"),
            y=alt.Y("avg_close", title="Avg Close"),
            color="ticker"
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.header("Avg Volume by Sector - Aggregation 2")
if agg2 is None or agg2.is_empty():
    st.info("agg2.parquet not found or empty. Run the data pipeline.")
else:
    df2 = agg2.to_pandas()
    # rename for nicer labels if necessary
    s_options = df2[df2.columns[0]].astype(str).unique().tolist()
    sel_sector = st.selectbox("Select sector", options=["All"] + s_options, index=0)
    if sel_sector != "All":
        df2 = df2[df2[df2.columns[0]] == sel_sector]
    # simple bar chart
    x_col, y_col = df2.columns[0], df2.columns[1]
    chart = alt.Chart(df2).mark_bar().encode(
        x=alt.X(y_col, title="Avg Volume"),
        y=alt.Y(x_col, sort="-x", title="Sector"),
    )
    st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.header("Daily Returns (Sample) - Aggregation 3")
if agg3 is None or agg3.is_empty():
    st.info("agg3.parquet not found or empty. Run the data pipeline.")
else:
    df3 = agg3.to_pandas()
    # convert date column
    date_cols = [c for c in df3.columns if "date" in c.lower() or c.lower() in ("dt", "day", "timestamp")]
    if date_cols:
        df3[date_cols[0]] = pd.to_datetime(df3[date_cols[0]])
    else:
        st.info("No date column found in agg3.")

    # allow ticker filtering
    t_cols = [c for c in df3.columns if "ticker" in c.lower() or c.lower() in ("symbol", "code")]
    if t_cols:
        tcol = t_cols[0]
        tick_options = sorted(df3[tcol].dropna().unique().tolist())
        sel_t = st.selectbox("Ticker to view returns", options=["All"] + tick_options, index=0)
        if sel_t != "All":
            df3 = df3[df3[tcol] == sel_t]

    # show table and histogram of daily_return
    if "daily_return" in df3.columns:
        st.subheader("Daily return histogram")
        df3 = df3.dropna(subset=["daily_return"])
        if not df3.empty:
            chart = alt.Chart(df3).mark_bar().encode(
                alt.X("daily_return", bin=alt.Bin(maxbins=50), title="Daily Return"),
                y='count()'
            )
            st.altair_chart(chart, use_container_width=True)
            st.subheader("Recent returns Table - Aggregation 3")
            st.dataframe(df3.sort_values(by=date_cols[0], ascending=False).head(100))
        else:
            st.write("No daily_return values available for selection.")
    else:
        st.info("Column 'daily_return' not found in agg3.")
