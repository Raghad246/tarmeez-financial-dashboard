import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date

st.set_page_config(page_title="Saudi Market Snapshot", layout="wide")

st.title("📈 Saudi Market Snapshot (TASI) — Live Financial Dashboard")
st.caption("Live financial dashboard built using Yahoo Finance data.")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

tickers = {
    "TASI Index": "^TASI.SR",
    "Al Rajhi Bank": "1120.SR",
    "Aramco": "2222.SR",
    "SABIC": "2010.SR",
    "STC": "7010.SR"
}

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    list(tickers.keys()),
    default=list(tickers.keys())
)

start_date = st.sidebar.date_input("Start date", date(2023, 1, 1))
end_date = st.sidebar.date_input("End date", date.today())

# -----------------------------
# Data Fetch
# -----------------------------
@st.cache_data
def load_data(selected, start, end):
    all_data = []
    for asset in selected:
        data = yf.download(tickers[asset], start=start, end=end)
        data["Asset"] = asset
        data = data.reset_index()
        all_data.append(data)
    return pd.concat(all_data)

if selected_assets:
    df = load_data(selected_assets, start_date, end_date)

    # -----------------------------
    # KPIs
    # -----------------------------
    latest_prices = df.groupby("Asset")["Close"].last()
    avg_price = latest_prices.mean()
    total_volume = df.groupby("Asset")["Volume"].sum().sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Assets Selected", len(selected_assets))
    col2.metric("Average Latest Price", f"{avg_price:,.2f} SAR")
    col3.metric("Total Volume", f"{int(total_volume):,}")

    st.divider()

    # -----------------------------
    # Price Trend Chart
    # -----------------------------
    fig = px.line(
        df,
        x="Date",
        y="Close",
        color="Asset",
        title="Price Trend",
        labels={"Close": "Closing Price (SAR)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Volume Chart
    # -----------------------------
    fig2 = px.bar(
        df,
        x="Date",
        y="Volume",
        color="Asset",
        title="Trading Volume Over Time"
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("Please select at least one asset.")
