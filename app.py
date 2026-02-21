import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# ------------------------------------------------
# Page Setup
# ------------------------------------------------
st.set_page_config(page_title="Saudi Market Snapshot (TASI)", layout="wide")

st.title("📈 Saudi Market Snapshot (TASI) — Executive Financial Dashboard")
st.caption("Live, decision-ready financial analytics dashboard. Built for investment analysis assessment.")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("Filters")

tickers = {
    "TASI Index (Benchmark)": "^TASI.SR",
    "Al Rajhi Bank": "1120.SR",
    "Aramco": "2222.SR",
    "SABIC": "2010.SR",
    "STC": "7010.SR",
}

assets = list(tickers.keys())

selected = st.sidebar.multiselect("Select Assets", assets, default=assets)

range_option = st.sidebar.selectbox("Quick Range", ["1M", "3M", "6M", "1Y", "YTD"])

today = date.today()

if range_option == "1M":
    start_date = today - timedelta(days=30)
elif range_option == "3M":
    start_date = today - timedelta(days=90)
elif range_option == "6M":
    start_date = today - timedelta(days=180)
elif range_option == "1Y":
    start_date = today - timedelta(days=365)
else:
    start_date = date(today.year, 1, 1)

end_date = today

risk_free = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, 0.25)
rolling_window = st.sidebar.slider("Rolling Window (days)", 10, 90, 30, 5)

if not selected:
    st.warning("Select at least one asset.")
    st.stop()

# ------------------------------------------------
# Data Download
# ------------------------------------------------
@st.cache_data
def load_data(selected, start, end):
    all_data = []
    for asset in selected:
        data = yf.download(tickers[asset], start=start, end=end, progress=False)
        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data = data.reset_index()
        data["Asset"] = asset
        all_data.append(data[["Date", "Close", "Volume", "Asset"]])

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data)

df = load_data(selected, start_date, end_date)

if df.empty:
    st.error("No data available for selected range.")
    st.stop()

prices = df.pivot(index="Date", columns="Asset", values="Close").dropna(how="all")
returns = prices.pct_change().dropna()

rf_daily = (risk_free/100)/252

# ------------------------------------------------
# Core Metrics
# ------------------------------------------------
latest_prices = prices.iloc[-1]
total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
volatility = returns.std() * np.sqrt(252) * 100
sharpe = ((returns.mean() - rf_daily) / returns.std()) * np.sqrt(252)
max_dd = ((prices / prices.cummax()) - 1).min() * 100
correlation = returns.corr()

# VaR / CVaR
def var_cvar(series):
    q = np.quantile(series.dropna(), 0.05)
    cvar = series[series <= q].mean()
    return q*100, cvar*100

var95 = {}
cvar95 = {}

for col in returns.columns:
    v, cv = var_cvar(returns[col])
    var95[col] = v
    cvar95[col] = cv

summary = pd.DataFrame({
    "Total Return %": total_return,
    "Volatility %": volatility,
    "Sharpe": sharpe,
    "Max Drawdown %": max_dd,
    "VaR 95% (daily)": pd.Series(var95),
    "CVaR 95% (daily)": pd.Series(cvar95),
}).round(2)

# ------------------------------------------------
# KPI Row
# ------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Assets Selected", len(selected))
col2.metric("Average Latest Price (SAR)", f"{latest_prices.mean():,.2f}")
col3.metric("Best Performer", total_return.idxmax())
col4.metric("Worst Performer", total_return.idxmin())

st.markdown("---")

# ------------------------------------------------
# Charts
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Risk", "Portfolio"])

# Overview
with tab1:
    fig_price = px.line(prices, title="Price Trend")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Executive Snapshot")
    st.dataframe(summary, use_container_width=True)

# Performance
with tab2:
    normalized = prices / prices.iloc[0] * 100
    fig_norm = px.line(normalized, title="Normalized Performance (Base = 100)")
    st.plotly_chart(fig_norm, use_container_width=True)

# Risk
with tab3:
    st.subheader("Correlation Matrix (Diversification Insight)")
    fig_corr = px.imshow(correlation, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    highest_corr = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack().idxmax()
    lowest_corr = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack().idxmin()

    st.write(f"🔎 Highest correlation pair: **{highest_corr}**")
    st.write(f"🧩 Best diversification pair (lowest correlation): **{lowest_corr}**")

    fig_vol = px.bar(summary, y="Volatility %", title="Annualized Volatility")
    st.plotly_chart(fig_vol, use_container_width=True)

# Portfolio (Random Efficient Frontier)
with tab4:
    st.subheader("Efficient Frontier Simulation")

    if len(returns.columns) < 2:
        st.warning("Select at least two assets.")
    else:
        mean = returns.mean()
        cov = returns.cov()

        sims = 1500
        results = []

        for _ in range(sims):
            weights = np.random.random(len(mean))
            weights /= weights.sum()

            ret = np.sum(mean * weights) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov*252, weights)))
            sharpe_ratio = (ret - risk_free/100) / vol

            results.append([ret*100, vol*100, sharpe_ratio])

        results = pd.DataFrame(results, columns=["Return %", "Volatility %", "Sharpe"])

        fig_frontier = px.scatter(results, x="Volatility %", y="Return %", color="Sharpe")
        st.plotly_chart(fig_frontier, use_container_width=True)

# ------------------------------------------------
# Decision-Ready Insights
# ------------------------------------------------
st.markdown("---")
st.subheader("📊 Decision-Ready Insights")

best_asset = summary["Total Return %"].idxmax()
lowest_risk = summary["Volatility %"].idxmin()
best_sharpe = summary["Sharpe"].idxmax()

st.write(f"📈 Growth-focused investor → Consider **{best_asset}** (highest total return).")
st.write(f"🛡️ Conservative investor → Consider **{lowest_risk}** (lowest volatility).")
st.write(f"⚖️ Risk-adjusted strategy → **{best_sharpe}** offers best Sharpe ratio.")

st.info("This dashboard provides quantitative insights for evaluation purposes. Not investment advice.")

# ------------------------------------------------
# Data Quality Section
# ------------------------------------------------
st.markdown("---")
st.subheader("Data Quality Overview")

coverage = prices.notna().sum().to_frame("Available Data Points")
st.dataframe(coverage)

csv = summary.to_csv().encode()
st.download_button("Download Executive Summary CSV", csv, "summary.csv", "text/csv")
