import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# -----------------------------
# Page config + subtle styling
# -----------------------------
st.set_page_config(page_title="Saudi Market Snapshot (TASI) — Executive", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid rgba(0,0,0,0.06); padding: 14px 14px; border-radius: 16px;}
    .chip {display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid rgba(0,0,0,0.08); background:#fff; margin-right:8px; font-size: 12px;}
    .muted {color: rgba(0,0,0,0.6); font-size: 13px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📈 Saudi Market Snapshot (TASI) — Executive Financial Dashboard")
st.caption("Live, decision-ready market analytics dashboard powered by Yahoo Finance. Built for assessment.")

# -----------------------------
# Sidebar (Filters)
# -----------------------------
st.sidebar.header("Filters")

TICKERS = {
    "TASI Index (Benchmark)": "^TASI.SR",
    "Al Rajhi Bank": "1120.SR",
    "Aramco": "2222.SR",
    "SABIC": "2010.SR",
    "STC": "7010.SR",
}

default_assets = list(TICKERS.keys())

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    options=list(TICKERS.keys()),
    default=default_assets,
)

preset = st.sidebar.selectbox(
    "Quick Range",
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

st.sidebar.divider()
risk_free = st.sidebar.slider("Risk-free rate (annual, %)", min_value=0.0, max_value=10.0, value=4.0, step=0.25)
rolling_window = st.sidebar.slider("Rolling Window (days)", min_value=10, max_value=120, value=30, step=5)

st.sidebar.divider()
st.sidebar.caption("Tip: If you see NaN in Beta/Alpha, increase date range or reduce rolling window.")

# Basic validation
if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()

if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

# Ensure benchmark presence (for beta/alpha)
bench_name = "TASI Index (Benchmark)"
if bench_name not in selected_assets:
    st.sidebar.warning("Benchmark (TASI) is required for Beta/Alpha. It will be added automatically.")
    selected_assets = [bench_name] + selected_assets

# -----------------------------
# Data fetch (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(assets, start, end):
    frames = []
    for a in assets:
        symbol = TICKERS[a]
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

        if data is None or data.empty:
            continue

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data.reset_index()
        data["Asset"] = a

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

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Date"])
    out = out.sort_values(["Date", "Asset"])
    return out

df = load_data(selected_assets, start_date, end_date)

if df.empty:
    st.warning("No data returned for the selected period. Try a different range.")
    st.stop()

# -----------------------------
# Transformations
# -----------------------------
df = df.dropna(subset=["Date"]).copy()
df = df.sort_values("Date")

prices = (
    df.pivot_table(index="Date", columns="Asset", values="Close", aggfunc="last")
      .sort_index()
      .dropna(how="all")
)

# Guard: ensure we have benchmark column
if bench_name not in prices.columns:
    st.error("Benchmark data (TASI) was not returned. Try another date range.")
    st.stop()

# Need enough data points
if prices.shape[0] < 5:
    st.warning("Not enough data points for analytics. Try a longer range.")
    st.stop()

returns = prices.pct_change().dropna(how="all")

# If too few returns
if returns.shape[0] < 3:
    st.warning("Not enough return points to compute risk metrics. Try a longer range.")
    st.stop()

# KPI base
latest_prices = prices.iloc[-1]
avg_latest_price = float(latest_prices.dropna().mean()) if latest_prices.dropna().shape[0] else np.nan
total_volume = float(df["Volume"].dropna().sum())
total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

# Annualized volatility
ann_vol = returns.std() * np.sqrt(252) * 100

# Max drawdown
cum = (1 + returns).cumprod()
drawdown = (cum / cum.cummax() - 1) * 100
max_drawdown = drawdown.min()

# Sharpe (annual)
rf_ann = risk_free / 100.0
rf_daily = (1 + rf_ann) ** (1/252) - 1
excess = returns.sub(rf_daily, axis=0)
ann_ret = ((1 + returns.mean()) ** 252 - 1) * 100
ann_excess_ret = ((1 + excess.mean()) ** 252 - 1) * 100
sharpe = (excess.mean() / returns.std()) * np.sqrt(252)

# -----------------------------
# Beta/Alpha vs TASI (ROBUST)
# -----------------------------
def beta_alpha_vs_benchmark(asset_returns: pd.Series, bench_returns: pd.Series):
    """
    Robust beta/alpha:
    - Align and drop NaNs
    - If insufficient points or benchmark variance ~0 => NaN
    """
    aligned = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if aligned.shape[0] < 10:
        return np.nan, np.nan

    x = aligned.iloc[:, 1].values  # benchmark
    y = aligned.iloc[:, 0].values  # asset
    var_x = np.var(x, ddof=1)
    if var_x <= 1e-12:
        return np.nan, np.nan

    cov_xy = np.cov(y, x, ddof=1)[0, 1]
    beta = cov_xy / var_x

    # Annualized alpha (CAPM-ish): alpha = (E[R]-Rf) - beta*(E[Rm]-Rf)
    mean_y = np.mean(y)
    mean_x = np.mean(x)
    alpha_daily = (mean_y - rf_daily) - beta * (mean_x - rf_daily)
    alpha_ann = ((1 + alpha_daily) ** 252 - 1) * 100
    return beta, alpha_ann

bench_ret = returns[bench_name].copy()

betas = {}
alphas = {}
excess_vs_tasi = {}

for col in returns.columns:
    b, a = beta_alpha_vs_benchmark(returns[col], bench_ret)
    betas[col] = b
    alphas[col] = a
    # Excess return vs benchmark (annualized difference)
    excess_vs_tasi[col] = (ann_ret[col] - ann_ret[bench_name]) if (col in ann_ret.index and bench_name in ann_ret.index) else np.nan

betas_s = pd.Series(betas, name="Beta vs TASI")
alphas_s = pd.Series(alphas, name="Alpha (ann) %")
excess_vs_tasi_s = pd.Series(excess_vs_tasi, name="Excess Return vs TASI %")

# Summary table
summary = pd.DataFrame({
    "Latest Price": latest_prices,
    "Total Return %": total_return,
    "Annual Return %": ann_ret,
    "Volatility % (ann)": ann_vol,
    "Sharpe (ann)": sharpe,
    "Max Drawdown %": max_drawdown,
    "Beta vs TASI": betas_s,
    "Alpha (ann) %": alphas_s,
    "Excess Return vs TASI %": excess_vs_tasi_s,
}).round(2)

summary = summary.sort_values("Total Return %", ascending=False)

# Normalized performance
normalized = prices.div(prices.iloc[0]).mul(100)

# Rolling metrics (robust min_periods)
minp = min(rolling_window, max(10, returns.shape[0]))
roll_vol = returns.rolling(rolling_window, min_periods=minp).std() * np.sqrt(252) * 100
roll_sharpe = (excess.rolling(rolling_window, min_periods=minp).mean() / returns.rolling(rolling_window, min_periods=minp).std()) * np.sqrt(252)

# Rolling beta vs benchmark
def rolling_beta(asset: pd.Series, bench: pd.Series, window: int, min_periods: int):
    aligned = pd.concat([asset, bench], axis=1)
    aligned.columns = ["asset", "bench"]
    def _beta(chunk):
        chunk = chunk.dropna()
        if chunk.shape[0] < min_periods:
            return np.nan
        x = chunk["bench"].values
        y = chunk["asset"].values
        var_x = np.var(x, ddof=1)
        if var_x <= 1e-12:
            return np.nan
        cov_xy = np.cov(y, x, ddof=1)[0, 1]
        return cov_xy / var_x
    return aligned.rolling(window).apply(lambda row: np.nan, raw=False)  # placeholder

# Manual rolling beta without pandas apply pitfalls:
roll_beta = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
for col in returns.columns:
    a = returns[col]
    b = bench_ret
    for i in range(len(returns.index)):
        start_i = max(0, i - rolling_window + 1)
        chunk = pd.concat([a.iloc[start_i:i+1], b.iloc[start_i:i+1]], axis=1).dropna()
        if chunk.shape[0] < minp:
            roll_beta.iloc[i, roll_beta.columns.get_loc(col)] = np.nan
            continue
        x = chunk.iloc[:, 1].values
        y = chunk.iloc[:, 0].values
        var_x = np.var(x, ddof=1)
        if var_x <= 1e-12:
            roll_beta.iloc[i, roll_beta.columns.get_loc(col)] = np.nan
        else:
            roll_beta.iloc[i, roll_beta.columns.get_loc(col)] = np.cov(y, x, ddof=1)[0, 1] / var_x

# -----------------------------
# Header chips
# -----------------------------
chips = [
    f"Risk-free: {risk_free:.2f}%",
    f"Rolling window: {rolling_window}d",
    f"Range: {preset}",
]
st.markdown(" ".join([f"<span class='chip'>{c}</span>" for c in chips]), unsafe_allow_html=True)

# -----------------------------
# Top KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets Selected", int(len(selected_assets)))
k2.metric("Average Latest Price (SAR)", f"{avg_latest_price:,.2f}" if np.isfinite(avg_latest_price) else "—")
k3.metric("Total Volume", f"{int(total_volume):,}" if np.isfinite(total_volume) else "—")
k4.metric("Period", f"{prices.index.min().date()} → {prices.index.max().date()}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_perf, tab_risk, tab_port, tab_data = st.tabs(
    ["Overview", "Performance", "Risk", "Portfolio", "Data"]
)

# -----------------------------
# Overview
# -----------------------------
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

    st.subheader("Quick Snapshot")
    st.dataframe(summary.reset_index().rename(columns={"index": "Asset"}), use_container_width=True, hide_index=True)

    st.subheader("Key Insights (Auto-generated)")
    # Best/Worst/Most risk + best Sharpe + top excess return
    best = summary["Total Return %"].idxmax()
    worst = summary["Total Return %"].idxmin()
    highest_risk = summary["Volatility % (ann)"].idxmax()
    best_sharpe = summary["Sharpe (ann)"].idxmax()
    top_excess = summary["Excess Return vs TASI %"].dropna().idxmax() if summary["Excess Return vs TASI %"].dropna().shape[0] else None

    st.write(f"✅ **Best performer (Total Return):** {best} ({summary.loc[best,'Total Return %']}%).")
    st.write(f"⚠️ **Worst performer (Total Return):** {worst} ({summary.loc[worst,'Total Return %']}%).")
    st.write(f"📌 **Highest volatility (Risk):** {highest_risk} ({summary.loc[highest_risk,'Volatility % (ann)']}%).")
    st.write(f"🏅 **Best risk-adjusted return (Sharpe):** {best_sharpe} ({summary.loc[best_sharpe,'Sharpe (ann)']}).")

    if top_excess is not None:
        st.write(f"📈 **Top outperformance vs TASI:** {top_excess} ({summary.loc[top_excess,'Excess Return vs TASI %']}% excess return).")

    st.markdown(
        "<div class='muted'>Volatility reflects price fluctuation (risk). Sharpe compares return per unit of risk. "
        "Correlation helps assess diversification. Beta/Alpha are measured relative to the benchmark (TASI).</div>",
        unsafe_allow_html=True
    )

# -----------------------------
# Performance
# -----------------------------
with tab_perf:
    st.subheader("Normalized Performance (Base = 100)")
    fig_norm = px.line(normalized, title="Normalized Performance (Base = 100)")
    st.plotly_chart(fig_norm, use_container_width=True)

    st.subheader(f"Rolling Sharpe ({rolling_window}d)")
    fig_rs = px.line(roll_sharpe, title=f"Rolling Sharpe ({rolling_window}d)")
    st.plotly_chart(fig_rs, use_container_width=True)

    st.subheader("Performance Summary (Top)")
    perf_cols = ["Total Return %", "Annual Return %", "Sharpe (ann)", "Latest Price"]
    st.dataframe(summary[perf_cols].sort_values("Sharpe (ann)", ascending=False).reset_index()
                 .rename(columns={"index": "Asset"}), use_container_width=True, hide_index=True)

# -----------------------------
# Risk
# -----------------------------
with tab_risk:
    c1, c2 = st.columns([1, 1])

    with c1:
        fig_vol = px.bar(
            summary.reset_index().rename(columns={"index": "Asset"}),
            x="Asset",
            y="Volatility % (ann)",
            title="Annualized Volatility (Risk)"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        fig_dd = px.bar(
            summary.reset_index().rename(columns={"index": "Asset"}),
            x="Asset",
            y="Max Drawdown %",
            title="Maximum Drawdown"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader(f"Rolling Volatility ({rolling_window}d)")
    fig_rv = px.line(roll_vol, title=f"Rolling Volatility ({rolling_window}d)")
    st.plotly_chart(fig_rv, use_container_width=True)

    st.subheader("Correlation Heatmap (Returns)")
    corr = returns.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Market Sensitivity vs TASI (Beta/Alpha)")
    ba_cols = ["Beta vs TASI", "Alpha (ann) %", "Excess Return vs TASI %"]
    st.dataframe(summary[ba_cols].reset_index().rename(columns={"index": "Asset"}),
                 use_container_width=True, hide_index=True)

    # Gentle warning if beta/alpha missing
    if summary["Beta vs TASI"].isna().any():
        st.info("Some Beta/Alpha values are NaN because there are not enough aligned data points for those assets in the selected range. Try a longer range or reduce rolling window.")

# -----------------------------
# Portfolio (simple optimizer without extra deps)
# -----------------------------
with tab_port:
    st.subheader("Portfolio Builder (No extra dependencies)")
    st.caption("Creates candidate portfolios using random weights (Dirichlet). Good for showcasing understanding without SciPy.")

    # pick investable assets (exclude benchmark for optimization)
    investable = [a for a in selected_assets if a != bench_name]
    if len(investable) < 2:
        st.warning("Select at least 2 non-benchmark assets to build portfolios.")
        st.stop()

    # returns subset
    R = returns[investable].dropna(how="all")
    if R.shape[0] < 20:
        st.warning("Not enough data for portfolio simulation. Try a longer range.")
        st.stop()

    mu_daily = R.mean().values
    cov_daily = R.cov().values

    n_ports = st.slider("Number of simulated portfolios", 500, 5000, 2000, 500)
    seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)

    np.random.seed(int(seed))
    W = np.random.dirichlet(alpha=np.ones(len(investable)), size=int(n_ports))

    # annualize
    mu_ann = (1 + mu_daily) ** 252 - 1
    rf = rf_ann
    port_ret = W @ mu_ann
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov_daily * 252, W))
    port_sharpe = (port_ret - rf) / np.where(port_vol == 0, np.nan, port_vol)

    ports = pd.DataFrame({
        "Return (ann)": port_ret * 100,
        "Volatility (ann)": port_vol * 100,
        "Sharpe": port_sharpe
    })

    # identify bests
    max_sharpe_idx = ports["Sharpe"].idxmax()
    min_vol_idx = ports["Volatility (ann)"].idxmin()

    st.markdown("**Suggested Portfolios**")
    s1, s2 = st.columns(2)

    with s1:
        st.success("Max Sharpe Portfolio")
        w_ms = pd.Series(W[max_sharpe_idx], index=investable).sort_values(ascending=False)
        st.dataframe(w_ms.rename("Weight").to_frame().reset_index().rename(columns={"index": "Asset"}), hide_index=True, use_container_width=True)
        st.write(f"Return (ann): **{ports.loc[max_sharpe_idx,'Return (ann)']:.2f}%**")
        st.write(f"Volatility (ann): **{ports.loc[max_sharpe_idx,'Volatility (ann)']:.2f}%**")
        st.write(f"Sharpe: **{ports.loc[max_sharpe_idx,'Sharpe']:.2f}**")

    with s2:
        st.info("Min Volatility Portfolio")
        w_mv = pd.Series(W[min_vol_idx], index=investable).sort_values(ascending=False)
        st.dataframe(w_mv.rename("Weight").to_frame().reset_index().rename(columns={"index": "Asset"}), hide_index=True, use_container_width=True)
        st.write(f"Return (ann): **{ports.loc[min_vol_idx,'Return (ann)']:.2f}%**")
        st.write(f"Volatility (ann): **{ports.loc[min_vol_idx,'Volatility (ann)']:.2f}%**")
        st.write(f"Sharpe: **{ports.loc[min_vol_idx,'Sharpe']:.2f}**")

    st.subheader("Efficient Frontier (Simulated)")
    fig_frontier = px.scatter(
        ports,
        x="Volatility (ann)",
        y="Return (ann)",
        color="Sharpe",
        title="Simulated Efficient Frontier",
        hover_data=["Sharpe"]
    )
    # markers for special portfolios
    fig_frontier.add_trace(go.Scatter(
        x=[ports.loc[max_sharpe_idx, "Volatility (ann)"]],
        y=[ports.loc[max_sharpe_idx, "Return (ann)"]],
        mode="markers",
        marker=dict(size=14, symbol="star"),
        name="Max Sharpe"
    ))
    fig_frontier.add_trace(go.Scatter(
        x=[ports.loc[min_vol_idx, "Volatility (ann)"]],
        y=[ports.loc[min_vol_idx, "Return (ann)"]],
        mode="markers",
        marker=dict(size=14, symbol="diamond"),
        name="Min Vol"
    ))
    st.plotly_chart(fig_frontier, use_container_width=True)

# -----------------------------
# Data tab
# -----------------------------
with tab_data:
    st.subheader("Raw Data (Sample)")
    st.dataframe(df.head(800), use_container_width=True)

    st.subheader("Download")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "market_data.csv", "text/csv")

    st.subheader("Notes")
    st.write("- If some tickers return missing volume, it will be handled as NA.")
    st.write("- Beta/Alpha require enough overlapping dates with the benchmark (TASI).")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Made with Streamlit • Data source: Yahoo Finance (via yfinance)")
