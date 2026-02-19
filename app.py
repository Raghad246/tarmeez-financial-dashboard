import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Saudi Market Snapshot (TASI)", layout="wide")

st.title("📈 Saudi Market Snapshot (TASI) — Live Financial Dashboard")
st.caption("Live, interactive financial dashboard powered by Yahoo Finance. Built for Tarmeez Capital assessment.")

# -----------------------------
# Sidebar
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

preset = st.sidebar.selectbox(
    "Quick range",
    ["Custom", "1M", "3M", "6M", "YTD", "1Y"],
    index=1
)

today = date.today()
if preset == "1M":
    start_date = today.replace(month=max(1, today.month-1))
elif preset == "3M":
    start_date = today.replace(month=max(1, today.month-3))
elif preset == "6M":
    start_date = today.replace(month=max(1, today.month-6))
elif preset == "YTD":
    start_date = date(today.year, 1, 1)
elif preset == "1Y":
    start_date = today.replace(year=today.year-1)
else:
    start_date = st.sidebar.date_input("Start date", date(2023, 1, 1))

end_date = st.sidebar.date_input("End date", today)

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(assets, start, end):
    frames = []
    for a in assets:
        df = yf.download(tickers[a], start=start, end=end, progress=False)
        if df.empty:
            continue
        df = df.reset_index()
        df["Asset"] = a
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

df = load_data(selected_assets, start_date, end_date)
if df.empty:
    st.warning("No data returned for the selected period.")
    st.stop()

# -----------------------------
# Transformations
# -----------------------------
prices = (
    df.pivot(index="Date", columns="Asset", values="Close")
      .sort_index()
      .dropna(how="all")
)

returns = prices.pct_change().dropna()

# KPIs
latest_prices = prices.iloc[-1]
avg_latest_price = latest_prices.mean()
total_volume = df.groupby("Asset")["Volume"].sum().sum()

total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
volatility = returns.std() * (252 ** 0.5) * 100  # annualized approx

cum = (1 + returns).cumprod()
drawdown = (cum / cum.cummax() - 1) * 100
max_drawdown = drawdown.min()

summary = (
    pd.DataFrame({
        "Total Return %": total_return.round(2),
        "Volatility %": volatility.round(2),
        "Max Drawdown %": max_drawdown.round(2),
        "Latest Price": latest_prices.round(2)
    })
    .sort_values("Total Return %", ascending=False)
)

# Normalized (Base=100)
normalized = prices / prices.iloc[0] * 100

# -----------------------------
# Top KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets Selected", len(selected_assets))
k2.metric("Average Latest Price (SAR)", f"{avg_latest_price:,.2f}")
k3.metric("Total Volume", f"{int(total_volume):,}")
k4.metric("Period", f"{prices.index.min().date()} → {prices.index.max().date()}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_perf, tab_risk, tab_data = st.tabs(
    ["Overview", "Performance", "Risk", "Data"]
)

# -------- Overview --------
with tab_overview:
    c1, c2 = st.columns([2, 1])

    with c1:
        fig_price = px.line(
            df,
            x="Date",
            y="Close",
            color="Asset",
            title="Price Trend",
            labels={"Close": "Closing Price (SAR)"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with c2:
        rank = latest_prices.sort_values(ascending=False).reset_index()
        rank.columns = ["Asset", "Latest Price"]
        fig_rank = px.bar(
            rank,
            x="Latest Price",
            y="Asset",
            orientation="h",
            title="Latest Price Ranking"
        )
        st.plotly_chart(fig_rank, use_container_width=True)

# -------- Performance --------
with tab_perf:
    fig_norm = px.line(
        normalized,
        title="Normalized Performance (Base = 100)"
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    st.subheader("Performance Summary")
    st.dataframe(summary[["Total Return %", "Latest Price"]], use_container_width=True)

# -------- Risk --------
with tab_risk:
    fig_vol = px.bar(
        summary.reset_index(),
        x="Asset",
        y="Volatility %",
        title="Annualized Volatility (Risk)"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    fig_dd = px.bar(
        summary.reset_index(),
        x="Asset",
        y="Max Drawdown %",
        title="Maximum Drawdown"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# -------- Data --------
with tab_data:
    st.subheader("Raw Data (Sample)")
    st.dataframe(df.head(500), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "market_data.csv", "text/csv")

# -----------------------------
# Insights
# -----------------------------
best = summary["Total Return %"].idxmax()
worst = summary["Total Return %"].idxmin()

st.divider()
st.subheader("Key Insights")
st.write(f"✅ **Best performer** over the selected period: **{best}** ({summary.loc[best,'Total Return %']}%).")
st.write(f"⚠️ **Worst performer**: **{worst}** ({summary.loc[worst,'Total Return %']}%).")
st.write(
    "📌 Assets with higher volatility exhibit higher risk. "
    "Normalized performance highlights relative outperformance independent of price levels."
)
