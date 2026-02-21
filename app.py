import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# =============================
# Page config + UI polish
# =============================
st.set_page_config(page_title="Saudi Market Snapshot (TASI)", layout="wide")

CUSTOM_CSS = """
<style>
/* Make the app feel cleaner */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 14px 14px 10px 14px;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 4px 18px rgba(0,0,0,0.04);
}
h1, h2, h3 { letter-spacing: -0.2px; }
.small-note { font-size: 0.92rem; color: rgba(0,0,0,0.6); }
.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.03);
    font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📈 Saudi Market Snapshot (TASI) — Live Financial Dashboard")
st.caption("Live, interactive dashboard powered by Yahoo Finance. Built for Tarmeez Capital assessment.")

# =============================
# Sidebar
# =============================
st.sidebar.header("Filters")

# Assets (Benchmark = TASI)
tickers = {
    "TASI Index (Benchmark)": "^TASI.SR",
    "Al Rajhi Bank": "1120.SR",
    "Aramco": "2222.SR",
    "SABIC": "2010.SR",
    "STC": "7010.SR"
}

default_assets = ["TASI Index (Benchmark)", "Al Rajhi Bank", "Aramco", "SABIC", "STC"]

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    list(tickers.keys()),
    default=default_assets
)

preset = st.sidebar.selectbox(
    "Quick range",
    ["1M", "3M", "6M", "YTD", "1Y", "Custom"],
    index=0
)

today = date.today()

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
    start_date = st.sidebar.date_input("Start date", date(today.year - 1, 1, 1))

end_date = st.sidebar.date_input("End date", today)

st.sidebar.markdown("---")

risk_free = st.sidebar.slider("Risk-free rate (annual, %)", 0.0, 10.0, 4.0, 0.25)
roll_window = st.sidebar.slider("Rolling window (days)", 10, 120, 30, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Portfolio Simulator**")
use_portfolio = st.sidebar.toggle("Enable portfolio (weights)", value=True)

# Validations
if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()

if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

# Ensure benchmark exists for advanced analytics
if "TASI Index (Benchmark)" not in selected_assets:
    st.info("Tip: Add **TASI Index (Benchmark)** to enable excess return, beta/alpha and benchmark comparison.")
    
# =============================
# Data fetch (robust + cached)
# =============================
@st.cache_data(show_spinner=False)
def load_data(asset_names, start, end):
    frames = []
    for a in asset_names:
        symbol = tickers[a]
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

        if data is None or data.empty:
            continue

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data.reset_index()
        data["Asset"] = a

        # Keep minimal required columns
        keep_cols = ["Date", "Close", "Volume", "Asset"]
        for col in keep_cols:
            if col not in data.columns:
                data[col] = pd.NA
        data = data[keep_cols].copy()

        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")

        frames.append(data)

    if not frames:
        return pd.DataFrame(columns=["Date", "Close", "Volume", "Asset"])

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df

with st.spinner("Fetching data..."):
    df = load_data(selected_assets, start_date, end_date)

if df.empty:
    st.warning("No data returned for the selected period. Try a different range.")
    st.stop()

# =============================
# Transformations
# =============================
prices = (
    df.pivot_table(index="Date", columns="Asset", values="Close", aggfunc="last")
      .sort_index()
      .dropna(how="all")
)

if prices.shape[0] < 5:
    st.warning("Not enough data points for robust analytics. Try a longer range (e.g., 3M or 6M).")
    st.stop()

returns = prices.pct_change().dropna(how="all")

# Helper metrics
latest_prices = prices.iloc[-1]
avg_latest_price = float(latest_prices.dropna().mean()) if latest_prices.dropna().shape[0] else np.nan
total_volume = float(df["Volume"].dropna().sum()) if df["Volume"].dropna().shape[0] else 0.0

total_return_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

# Annualized vol (252 trading days assumption)
ann_vol_pct = returns.std() * np.sqrt(252) * 100

# Drawdown
cum = (1 + returns).cumprod()
drawdown = (cum / cum.cummax() - 1) * 100
max_drawdown_pct = drawdown.min()

# Sharpe (annualized) - using daily rf
rf_daily = (risk_free / 100) / 252
excess_daily = returns.sub(rf_daily)
sharpe = (excess_daily.mean() / returns.std()) * np.sqrt(252)

# Summary table
summary = pd.DataFrame({
    "Latest Price": latest_prices.round(2),
    "Total Return %": total_return_pct.round(2),
    "Volatility % (ann)": ann_vol_pct.round(2),
    "Sharpe (ann)": sharpe.round(2),
    "Max Drawdown %": max_drawdown_pct.round(2),
}).sort_values("Total Return %", ascending=False)

# Normalized (Base=100)
normalized = prices.div(prices.iloc[0]).mul(100)

# Benchmark analytics (beta/alpha/excess)
benchmark_name = "TASI Index (Benchmark)"
has_bench = benchmark_name in returns.columns and returns[benchmark_name].dropna().shape[0] >= 10

beta = pd.Series(index=returns.columns, dtype=float)
alpha_ann = pd.Series(index=returns.columns, dtype=float)

if has_bench:
    bench = returns[benchmark_name].dropna()
    for col in returns.columns:
        if col == benchmark_name:
            beta[col] = 1.0
            alpha_ann[col] = 0.0
            continue
        aligned = pd.concat([returns[col], bench], axis=1).dropna()
        if aligned.shape[0] < 10:
            beta[col] = np.nan
            alpha_ann[col] = np.nan
            continue
        r_i = aligned.iloc[:, 0]
        r_m = aligned.iloc[:, 1]
        cov = np.cov(r_i, r_m, ddof=1)[0, 1]
        var = np.var(r_m, ddof=1)
        b = cov / var if var != 0 else np.nan
        beta[col] = b
        # annualized alpha approximation: (mean_i - rf) - beta*(mean_m - rf) then *252
        alpha_daily = (r_i.mean() - rf_daily) - b * (r_m.mean() - rf_daily)
        alpha_ann[col] = alpha_daily * 252

    summary["Beta vs TASI"] = beta.round(2)
    summary["Alpha (ann)"] = alpha_ann.round(2)

# Excess return vs benchmark
excess_return_pct = None
if has_bench:
    excess_return_pct = (total_return_pct - total_return_pct[benchmark_name]).round(2)

# Rolling volatility (annualized)
rolling_vol = returns.rolling(roll_window).std() * np.sqrt(252) * 100

# =============================
# KPIs Row
# =============================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets Selected", len(selected_assets))
k2.metric("Average Latest Price (SAR)", f"{avg_latest_price:,.2f}" if np.isfinite(avg_latest_price) else "—")
k3.metric("Total Volume", f"{int(total_volume):,}" if np.isfinite(total_volume) else "—")
k4.metric("Period", f"{prices.index.min().date()} → {prices.index.max().date()}")

st.markdown(
    f'<span class="badge">Risk-free: {risk_free:.2f}%</span> &nbsp; '
    f'<span class="badge">Rolling window: {roll_window}d</span>',
    unsafe_allow_html=True
)

st.divider()

# =============================
# Tabs
# =============================
tab_overview, tab_perf, tab_risk, tab_port, tab_data = st.tabs(
    ["Overview", "Performance", "Risk", "Portfolio", "Data"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2 = st.columns([2.2, 1])

    with c1:
        fig_price = px.line(
            df.dropna(subset=["Close"]),
            x="Date", y="Close", color="Asset",
            title="Price Trend",
            labels={"Close": "Closing Price (SAR)"}
        )
        fig_price.update_layout(legend_title_text="Asset", height=420)
        st.plotly_chart(fig_price, use_container_width=True)

    with c2:
        rank = latest_prices.dropna().sort_values(ascending=False).reset_index()
        rank.columns = ["Asset", "Latest Price"]
        fig_rank = px.bar(rank, x="Latest Price", y="Asset", orientation="h", title="Latest Price Ranking")
        fig_rank.update_layout(height=420)
        st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("### Quick Snapshot")
    snap = summary.copy()
    if has_bench:
        snap["Excess Return vs TASI %"] = excess_return_pct
    st.dataframe(snap, use_container_width=True)

# -----------------------------
# Performance
# -----------------------------
with tab_perf:
    st.subheader("Normalized Performance (Base = 100)")
    fig_norm = px.line(normalized, title="Normalized Performance (Base = 100)")
    fig_norm.update_layout(height=420)
    st.plotly_chart(fig_norm, use_container_width=True)

    if has_bench:
        st.subheader("Excess Return vs TASI (Cumulative, %)")
        # cumulative excess: (asset cum return - bench cum return)
        cum_ret = (prices / prices.iloc[0] - 1) * 100
        excess_cum = cum_ret.sub(cum_ret[benchmark_name], axis=0).drop(columns=[benchmark_name], errors="ignore")

        fig_ex = px.line(excess_cum, title="Excess Return vs TASI (Cumulative, %)")
        fig_ex.update_layout(height=420)
        st.plotly_chart(fig_ex, use_container_width=True)
    else:
        st.info("Add **TASI Index (Benchmark)** to see excess-return analytics.")

# -----------------------------
# Risk
# -----------------------------
with tab_risk:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Rolling Volatility (Annualized %)")
        fig_rv = px.line(rolling_vol, title=f"Rolling Volatility — {roll_window} Days")
        fig_rv.update_layout(height=420)
        st.plotly_chart(fig_rv, use_container_width=True)

    with c2:
        st.subheader("Max Drawdown (%) over time")
        fig_dd = px.line(drawdown, title="Drawdown (%)")
        fig_dd.update_layout(height=420)
        st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Correlation Matrix (Diversification)")
    corr = returns.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Risk Metrics Table")
    risk_cols = ["Volatility % (ann)", "Sharpe (ann)", "Max Drawdown %"]
    if has_bench:
        risk_cols += ["Beta vs TASI", "Alpha (ann)"]
    st.dataframe(summary[risk_cols].sort_values("Sharpe (ann)", ascending=False), use_container_width=True)

# -----------------------------
# Portfolio
# -----------------------------
with tab_port:
    if not use_portfolio:
        st.info("Portfolio simulator is disabled from sidebar.")
    else:
        st.subheader("Portfolio Simulator (Weights)")

        # Choose investable assets (exclude benchmark by default from portfolio)
        investable = [a for a in selected_assets if a != benchmark_name]
        if len(investable) < 1:
            st.warning("Select at least one non-benchmark asset to build a portfolio.")
            st.stop()

        st.caption("Set weights — the app will auto-normalize if the sum is not 100%.")

        weights = {}
        cols = st.columns(min(5, len(investable)))
        for i, a in enumerate(investable):
            with cols[i % len(cols)]:
                weights[a] = st.slider(a, 0, 100, int(100 / len(investable)))

        w = pd.Series(weights, dtype=float)
        if w.sum() == 0:
            st.warning("Set at least one weight above 0%.")
            st.stop()
        w = w / w.sum()

        # Build portfolio returns (aligned)
        port_returns = returns[investable].dropna(how="all").fillna(0).dot(w)
        port_prices = (1 + port_returns).cumprod()

        # Benchmark comparison
        if has_bench:
            bench_ret = returns[benchmark_name].reindex(port_returns.index).fillna(0)
            bench_cum = (1 + bench_ret).cumprod()
        else:
            bench_cum = None

        # Portfolio metrics
        port_total = (port_prices.iloc[-1] / port_prices.iloc[0] - 1) * 100
        port_vol = port_returns.std() * np.sqrt(252) * 100
        port_sharpe = ((port_returns.mean() - rf_daily) / port_returns.std()) * np.sqrt(252) if port_returns.std() != 0 else np.nan

        # Drawdown
        port_cum = (1 + port_returns).cumprod()
        port_dd = (port_cum / port_cum.cummax() - 1) * 100
        port_mdd = port_dd.min()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Portfolio Total Return %", f"{port_total:.2f}")
        m2.metric("Portfolio Volatility % (ann)", f"{port_vol:.2f}")
        m3.metric("Portfolio Sharpe (ann)", f"{port_sharpe:.2f}" if np.isfinite(port_sharpe) else "—")
        m4.metric("Portfolio Max Drawdown %", f"{port_mdd:.2f}")

        # Chart: portfolio vs benchmark
        chart_df = pd.DataFrame({"Portfolio": port_cum * 100})
        if bench_cum is not None:
            chart_df["TASI (Benchmark)"] = bench_cum * 100

        fig_port = px.line(chart_df, title="Portfolio vs Benchmark (Indexed)")
        fig_port.update_layout(height=440)
        st.plotly_chart(fig_port, use_container_width=True)

        fig_pdd = px.line(port_dd, title="Portfolio Drawdown (%)")
        fig_pdd.update_layout(height=360)
        st.plotly_chart(fig_pdd, use_container_width=True)

        st.subheader("Weights")
        w_df = (w * 100).round(2).reset_index()
        w_df.columns = ["Asset", "Weight %"]
        st.dataframe(w_df, use_container_width=True)

# -----------------------------
# Data
# -----------------------------
with tab_data:
    st.subheader("Raw Data (Sample)")
    st.dataframe(df.head(800), use_container_width=True)

    st.subheader("Download")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Raw CSV", csv, "market_data.csv", "text/csv")

    summary_csv = summary.reset_index().rename(columns={"index": "Asset"}).to_csv(index=False).encode("utf-8")
    st.download_button("Download Metrics CSV", summary_csv, "metrics_summary.csv", "text/csv")

# =============================
# Key Insights (judge-friendly)
# =============================
st.divider()
st.subheader("Key Insights (Auto-generated)")

best = summary["Total Return %"].idxmax()
worst = summary["Total Return %"].idxmin()
highest_risk = summary["Volatility % (ann)"].idxmax()
best_sharpe = summary["Sharpe (ann)"].idxmax()

st.write(f"✅ **Best performer (Total Return):** **{best}** ({summary.loc[best,'Total Return %']}%).")
st.write(f"⚠️ **Worst performer (Total Return):** **{worst}** ({summary.loc[worst,'Total Return %']}%).")
st.write(f"📌 **Highest volatility (Risk):** **{highest_risk}** ({summary.loc[highest_risk,'Volatility % (ann)']}%).")
st.write(f"🏅 **Best risk-adjusted return (Sharpe):** **{best_sharpe}** ({summary.loc[best_sharpe,'Sharpe (ann)']}).")

if has_bench:
    # Outperformance vs benchmark
    non_bench = summary.drop(index=[benchmark_name], errors="ignore").copy()
    non_bench["Excess Return vs TASI %"] = excess_return_pct.drop(index=[benchmark_name], errors="ignore")
    top_excess = non_bench["Excess Return vs TASI %"].idxmax()
    st.write(
        f"📈 **Top outperformance vs TASI:** **{top_excess}** "
        f"({non_bench.loc[top_excess,'Excess Return vs TASI %']}% excess return)."
    )

    # Beta insight
    if "Beta vs TASI" in summary.columns:
        highest_beta = summary["Beta vs TASI"].dropna().idxmax()
        st.write(
            f"🧠 **Market sensitivity:** **{highest_beta}** has the highest **Beta** vs TASI "
            f"({summary.loc[highest_beta,'Beta vs TASI']})."
        )

st.markdown(
    '<div class="small-note">ℹ️ Volatility reflects price fluctuation (risk). '
    'Sharpe compares return per unit of risk. Correlation helps assess diversification. '
    'Beta/Alpha are measured relative to the benchmark (TASI).</div>',
    unsafe_allow_html=True
)
