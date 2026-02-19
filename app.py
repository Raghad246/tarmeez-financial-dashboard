import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta

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

def safe_month_back(d: date, months_back: int) -> date:
    # Simple month-back helper without extra dependencies
    y, m = d.year, d.month - months_back
    while m <= 0:
        y -= 1
        m += 12
    # clamp day
    day = min(d.day, 28)
    return date(y, m, day)

if preset == "1M":
    start_date = today - timedelta(days=30)
elif preset == "3M":
    start_date = today - timedelta(days=90)
elif preset == "6M":
    start_date = today - timedelta(days=180)
elif preset == "YTD":
    start_date = date(today.year, 1, 1)
elif preset == "1Y":
    start_date = today - timedelta(days=365)
else:
    start_date = st.sidebar.date_input("Start date", date(2023, 1, 1))

end_date = st.sidebar.date_input("End date", today)

# Basic validation
if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()

if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(assets, start, end):
    frames = []
    for a in assets:
        symbol = tickers[a]
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

        if data is None or data.empty:
            continue

        # Flatten MultiIndex columns if present (yfinance sometimes returns this)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data.reset_index()
        data["Asset"] = a

        # Keep only needed columns (some tickers may not have Volume consistently)
        keep_cols = ["Date", "Close", "Volume", "Asset"]
        for col in keep_cols:
            if col not in data.columns:
                data[col] = pd.NA

        data = data[keep_cols].copy()

        # Ensure Close/Volume are 1D numeric
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")

        frames.append(data)

    if not frames:
        return pd.DataFrame(columns=["Date", "Close", "Volume", "Asset"])

    return pd.concat(frames, ignore_index=True)

df = load_data(selected_assets, start_date, end_date)

if df.empty:
    st.warning("No data returned for the selected period. Try a different range.")
    st.stop()

# -----------------------------
# Transformations
# -----------------------------
df = df.dropna(subset=["Date"]).copy()
df = df.sort_values("Date")

# Prices matrix (safe pivot_table)
prices = (
    df.pivot_table(index="Date", columns="Asset", values="Close", aggfunc="last")
      .sort_index()
      .dropna(how="all")
)

# If we still don't have enough data for returns:
if prices.shape[0] < 3:
    st.warning("Not enough data points in this period to compute returns. Try a longer range.")
    st.stop()

returns = prices.pct_change().dropna(how="all")

# KPIs
latest_prices = prices.iloc[-1]
avg_latest_price = latest_prices.mean()

total_volume = df["Volume"].dropna().sum()
total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

# Annualized volatility (approx, assumes ~252 trading days)
volatility = returns.std() * (252 ** 0.5) * 100

# Max drawdown
cum = (1 + returns).cumprod()
dd = (cum / cum.cummax() - 1) * 100
max_drawdown = dd.min()

summary = (
    pd.DataFrame({
        "Total Return %": total_return.round(2),
        "Volatility %": volatility.round(2),
        "Max Drawdown %": max_drawdown.round(2),
        "Latest Price": latest_prices.round(2),
    })
    .sort_values("Total Return %", ascending=False)
)

# Normalized performance (Base=100)
normalized = prices.div(prices.iloc[0]).mul(100)

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
            df.dropna(subset=["Close"]),
            x="Date",
            y="Close",
            color="Asset",
            title="Price Trend",
            labels={"Close": "Closing Price (SAR)"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with c2:
        rank = latest_prices.dropna().sort_values(ascending=False).reset_index()
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
        summary.reset_index().rename(columns={"index": "Asset"}),
        x="Asset",
        y="Volatility %",
        title="Annualized Volatility (Risk)"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    fig_dd = px.bar(
        summary.reset_index().rename(columns={"index": "Asset"}),
        x="Asset",
        y="Max Drawdown %",
        title="Maximum Drawdown"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Risk Summary")
    st.dataframe(summary[["Volatility %", "Max Drawdown %"]], use_container_width=True)

# -------- Data --------
with tab_data:
    st.subheader("Raw Data (Sample)")
    st.dataframe(df.head(500), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "market_data.csv", "text/csv")

# -----------------------------
# Insights
# -----------------------------
st.divider()
st.subheader("Key Insights")

best = summary["Total Return %"].idxmax()
worst = summary["Total Return %"].idxmin()

st.write(f"✅ **Best performer** over the selected period: **{best}** ({summary.loc[best,'Total Return %']}%).")
st.write(f"⚠️ **Worst performer**: **{worst}** ({summary.loc[worst,'Total Return %']}%).")

highest_risk = summary["Volatility %"].idxmax()
st.write(f"📌 **Highest volatility (risk)**: **{highest_risk}** ({summary.loc[highest_risk,'Volatility %']}%).")

st.write(
    "ℹ️ Volatility reflects price fluctuation (risk). Normalized performance compares assets fairly regardless of price levels."
)
