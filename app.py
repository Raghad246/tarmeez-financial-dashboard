import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Saudi Market Snapshot (TASI)",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Saudi Market Snapshot (TASI) — Executive Financial Dashboard")
st.caption("Live, decision-ready market analytics powered by Yahoo Finance. Built for assessment.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")

TICKERS = {
    "TASI Index (Benchmark)": "^TASI.SR",
    "Al Rajhi Bank": "1120.SR",
    "Aramco": "2222.SR",
    "SABIC": "2010.SR",
    "STC": "7010.SR",
}

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    list(TICKERS.keys()),
    default=list(TICKERS.keys()),
)

preset = st.sidebar.selectbox(
    "Quick Range",
    ["1M", "3M", "6M", "YTD", "1Y", "Custom"],
    index=0
)

today = date.today()

def preset_to_start(p: str) -> date:
    if p == "1M":
        return today - timedelta(days=30)
    if p == "3M":
        return today - timedelta(days=90)
    if p == "6M":
        return today - timedelta(days=180)
    if p == "YTD":
        return date(today.year, 1, 1)
    if p == "1Y":
        return today - timedelta(days=365)
    return date(2023, 1, 1)

if preset == "Custom":
    start_date = st.sidebar.date_input("Start date", date(2023, 1, 1))
else:
    start_date = preset_to_start(preset)

end_date = st.sidebar.date_input("End date", today)

st.sidebar.divider()
risk_free = st.sidebar.slider("Risk-free rate (annual, %)", 0.0, 10.0, 4.0, 0.25)

# Validation
if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(show_spinner=False)
def load_prices(asset_names, start, end):
    frames = []
    for name in asset_names:
        symbol = TICKERS[name]
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

        if data is None or data.empty:
            continue

        # yfinance may return MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data.reset_index()
        data["Asset"] = name

        # Ensure columns exist
        for col in ["Date", "Close", "Volume"]:
            if col not in data.columns:
                data[col] = pd.NA

        data = data[["Date", "Close", "Volume", "Asset"]].copy()
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
        frames.append(data)

    if not frames:
        return pd.DataFrame(columns=["Date", "Close", "Volume", "Asset"])

    df_ = pd.concat(frames, ignore_index=True)
    df_ = df_.dropna(subset=["Date"]).sort_values(["Date", "Asset"])
    return df_

df = load_prices(selected_assets, start_date, end_date)

if df.empty:
    st.warning("No data returned for the selected period. Try a different range.")
    st.stop()

# -----------------------------
# Transformations
# -----------------------------
prices = (
    df.pivot_table(index="Date", columns="Asset", values="Close", aggfunc="last")
      .sort_index()
      .dropna(how="all")
)

if prices.shape[0] < 3:
    st.warning("Not enough data points to compute returns. Try a longer range.")
    st.stop()

returns = prices.pct_change().dropna(how="all")

bench_name = "TASI Index (Benchmark)" if "TASI Index (Benchmark)" in prices.columns else None

def safe_beta_alpha(asset_ret: pd.Series, bench_ret: pd.Series):
    aligned = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
    if aligned.shape[0] < 5:
        return (np.nan, np.nan, np.nan)

    a = aligned.iloc[:, 0].astype(float)
    b = aligned.iloc[:, 1].astype(float)

    var_b = float(np.var(b, ddof=1))
    if var_b == 0.0 or np.isnan(var_b):
        return (np.nan, np.nan, np.nan)

    cov_ab = float(np.cov(a, b, ddof=1)[0, 1])
    beta_val = cov_ab / var_b

    alpha_daily = float(a.mean() - beta_val * b.mean())
    alpha_ann = alpha_daily * 252 * 100

    asset_total = (1 + a).prod() - 1
    bench_total = (1 + b).prod() - 1
    excess = (asset_total - bench_total) * 100

    return (beta_val, alpha_ann, excess)

# KPIs
latest_prices = prices.iloc[-1]
avg_latest_price = float(latest_prices.dropna().mean()) if latest_prices.dropna().shape[0] else np.nan
total_volume = float(df["Volume"].dropna().sum())
period_label = f"{prices.index.min().date()} → {prices.index.max().date()}"

# Metrics
total_return_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
ann_return_pct = ((1 + returns.mean()) ** 252 - 1) * 100
volatility_pct = returns.std() * np.sqrt(252) * 100

cum = (1 + returns).cumprod()
dd = (cum / cum.cummax() - 1) * 100
max_drawdown_pct = dd.min()

rf_daily = (risk_free / 100) / 252
excess_daily = returns.sub(rf_daily)
sharpe_ann = (excess_daily.mean() / excess_daily.std()) * np.sqrt(252)

beta = pd.Series(index=prices.columns, dtype="float64")
alpha_ann = pd.Series(index=prices.columns, dtype="float64")
excess_vs_tasi = pd.Series(index=prices.columns, dtype="float64")

if bench_name is not None and bench_name in returns.columns:
    bench_ret = returns[bench_name]
    for col in returns.columns:
        b, a, ex = safe_beta_alpha(returns[col], bench_ret)
        beta[col] = b
        alpha_ann[col] = a
        excess_vs_tasi[col] = ex

summary = pd.DataFrame({
    "Latest Price": latest_prices,
    "Total Return %": total_return_pct,
    "Annual Return %": ann_return_pct,
    "Volatility % (ann)": volatility_pct,
    "Sharpe (ann)": sharpe_ann,
    "Max Drawdown %": max_drawdown_pct,
})

if bench_name is not None:
    summary["Beta vs TASI"] = beta
    summary["Alpha (ann) %"] = alpha_ann
    summary["Excess Return vs TASI %"] = excess_vs_tasi

summary = summary.replace([np.inf, -np.inf], np.nan)

# Normalized performance
normalized = prices.div(prices.iloc[0]).mul(100)

# -----------------------------
# KPI Cards
# -----------------------------
chip1, chip2, chip3 = st.columns([1, 1, 1])
with chip1:
    st.markdown(f"**Risk-free:** {risk_free:.2f}%")
with chip2:
    st.markdown(f"**Range:** {preset}")
with chip3:
    st.markdown(f"**Benchmark:** {'TASI' if bench_name else '—'}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets Selected", int(len(selected_assets)))
k2.metric("Average Latest Price (SAR)", "-" if np.isnan(avg_latest_price) else f"{avg_latest_price:,.2f}")
k3.metric("Total Volume", f"{int(total_volume):,}" if not np.isnan(total_volume) else "-")
k4.metric("Period", period_label)

st.divider()

# -----------------------------
# Tabs (Clean)
# -----------------------------
tab_overview, tab_perf, tab_risk = st.tabs(["Overview", "Performance", "Risk"])

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
            labels={"Close": "Closing Price (SAR)"},
        )
        fig_price.update_layout(legend_title_text="Asset")
        st.plotly_chart(fig_price, use_container_width=True)

    with c2:
        rank = latest_prices.dropna().sort_values(ascending=False).reset_index()
        rank.columns = ["Asset", "Latest Price"]
        fig_rank = px.bar(
            rank,
            x="Latest Price",
            y="Asset",
            orientation="h",
            title="Latest Price Ranking",
        )
        fig_rank.update_layout(yaxis_title=None)
        st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("Quick Snapshot")
    show_cols = [
        "Latest Price", "Total Return %", "Annual Return %", "Volatility % (ann)",
        "Sharpe (ann)", "Max Drawdown %"
    ]
    if bench_name is not None:
        show_cols += ["Beta vs TASI", "Alpha (ann) %", "Excess Return vs TASI %"]

    snap = summary[show_cols].copy().round(2)
    snap = snap.reset_index().rename(columns={"index": "Asset"})
    st.dataframe(snap, use_container_width=True, hide_index=True)

# -------- Performance --------
with tab_perf:
    st.subheader("Normalized Performance (Base = 100)")
    fig_norm = px.line(normalized, title="")
    fig_norm.update_layout(legend_title_text="Asset")
    st.plotly_chart(fig_norm, use_container_width=True)

    st.subheader("Performance Summary (Top)")
    perf_tbl = summary[["Total Return %", "Annual Return %", "Sharpe (ann)", "Latest Price"]].copy()
    perf_tbl = perf_tbl.sort_values("Total Return %", ascending=False).round(2)
    perf_tbl = perf_tbl.reset_index().rename(columns={"index": "Asset"})
    st.dataframe(perf_tbl, use_container_width=True, hide_index=True)

# -------- Risk --------
with tab_risk:
    c1, c2 = st.columns(2)

    with c1:
        vol_plot = summary["Volatility % (ann)"].dropna().sort_values(ascending=False).reset_index()
        vol_plot.columns = ["Asset", "Volatility % (ann)"]
        fig_vol = px.bar(vol_plot, x="Asset", y="Volatility % (ann)", title="Annualized Volatility (Risk)")
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        dd_plot = summary["Max Drawdown %"].dropna().sort_values().reset_index()
        dd_plot.columns = ["Asset", "Max Drawdown %"]
        fig_dd = px.bar(dd_plot, x="Asset", y="Max Drawdown %", title="Maximum Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Correlation Heatmap (Returns)")
    corr = returns.corr()
    if not corr.empty and not corr.isna().all().all():
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                zmin=-1, zmax=1,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>",
            )
        )
        fig_corr.update_layout(title="Correlation Matrix", height=520)
        st.plotly_chart(fig_corr, use_container_width=True)

    if bench_name is not None and "Beta vs TASI" in summary.columns:
        st.subheader("Market Sensitivity vs TASI (Beta / Alpha)")
        ba = summary[["Beta vs TASI", "Alpha (ann) %", "Excess Return vs TASI %"]].copy().round(2)
        ba = ba.reset_index().rename(columns={"index": "Asset"})
        st.dataframe(ba, use_container_width=True, hide_index=True)

# -----------------------------
# Insights (Auto)
# -----------------------------
st.divider()
st.subheader("Key Insights (Auto-generated)")

safe_summary = summary.copy()

best = safe_summary["Total Return %"].idxmax()
worst = safe_summary["Total Return %"].idxmin()
highest_risk = safe_summary["Volatility % (ann)"].idxmax()
best_sharpe = safe_summary["Sharpe (ann)"].idxmax()

st.write(f"✅ **Best performer (Total Return):** **{best}** ({safe_summary.loc[best, 'Total Return %']:.2f}%).")
st.write(f"⚠️ **Worst performer (Total Return):** **{worst}** ({safe_summary.loc[worst, 'Total Return %']:.2f}%).")
st.write(f"📌 **Highest volatility (Risk):** **{highest_risk}** ({safe_summary.loc[highest_risk, 'Volatility % (ann)']:.2f}%).")
if not np.isnan(safe_summary.loc[best_sharpe, "Sharpe (ann)"]):
    st.write(f"🏅 **Best risk-adjusted return (Sharpe):** **{best_sharpe}** ({safe_summary.loc[best_sharpe, 'Sharpe (ann)']:.2f}).")

if bench_name is not None and "Excess Return vs TASI %" in safe_summary.columns:
    top_excess = safe_summary["Excess Return vs TASI %"].idxmax()
    val = safe_summary.loc[top_excess, "Excess Return vs TASI %"]
    if not np.isnan(val):
        st.write(f"📈 **Top outperformance vs TASI:** **{top_excess}** ({val:.2f}% excess return).")

st.info(
    "Volatility reflects price fluctuation (risk). Sharpe compares return per unit of risk. "
    "Correlation helps assess diversification. Beta/Alpha are measured relative to the benchmark (TASI)."
)
