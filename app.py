import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

# -----------------------------
# Page config + light styling
# -----------------------------
st.set_page_config(page_title="Saudi Market Snapshot (TASI)", layout="wide")

st.markdown(
    """
    <style>
      .small-note {color:#6b7280; font-size:0.9rem;}
      .kpi-card {border:1px solid #eef0f3; padding:16px; border-radius:16px; background:white;}
      .kpi-label {color:#6b7280; font-size:0.85rem; margin-bottom:6px;}
      .kpi-value {font-size:1.65rem; font-weight:700;}
      .pill {display:inline-block; padding:6px 10px; border:1px solid #eef0f3; border-radius:999px; background:#fafafa; margin-right:8px; font-size:0.85rem;}
      .divider {height:1px; background:#eef0f3; margin:16px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Saudi Market Snapshot (TASI) — Live Financial Dashboard")
st.caption("Live, interactive financial dashboard powered by Yahoo Finance. Built for Tarmeez Capital assessment.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")

tickers = {
    "TASI Index (Benchmark)": "^TASI.SR",
    "Al Rajhi Bank": "1120.SR",
    "Aramco": "2222.SR",
    "SABIC": "2010.SR",
    "STC": "7010.SR",
}

asset_names = list(tickers.keys())

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    asset_names,
    default=asset_names,
)

preset = st.sidebar.selectbox(
    "Quick range",
    ["Custom", "1M", "3M", "6M", "YTD", "1Y"],
    index=1,
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
    start_date = st.sidebar.date_input("Start date", date(2023, 1, 1))

end_date = st.sidebar.date_input("End date", today)

st.sidebar.markdown("---")
risk_free = st.sidebar.slider("Risk-free rate (annual, %)", 0.0, 10.0, 4.0, 0.25)
roll_window = st.sidebar.slider("Rolling window (days)", 10, 90, 30, 5)

if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    st.stop()

if not selected_assets:
    st.warning("Please select at least one asset.")
    st.stop()

benchmark_name = "TASI Index (Benchmark)"
if benchmark_name not in selected_assets:
    st.sidebar.info("Tip: Include TASI benchmark for Beta/Alpha metrics.")

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(selected: list, start: date, end: date) -> pd.DataFrame:
    frames = []
    for a in selected:
        symbol = tickers[a]
        try:
            data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        except Exception:
            data = pd.DataFrame()

        if data is None or data.empty:
            continue

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data.reset_index()
        data["Asset"] = a

        # Ensure expected cols exist
        for col in ["Date", "Close", "Volume"]:
            if col not in data.columns:
                data[col] = np.nan

        data = data[["Date", "Close", "Volume", "Asset"]].copy()
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

df = df.dropna(subset=["Date"]).sort_values("Date").copy()

prices = (
    df.pivot_table(index="Date", columns="Asset", values="Close", aggfunc="last")
      .sort_index()
      .dropna(how="all")
)

if prices.shape[0] < 3:
    st.warning("Not enough data points in this period. Try a longer range.")
    st.stop()

returns = prices.pct_change().dropna(how="all")
rf_daily = (risk_free / 100) / 252.0

latest_prices = prices.iloc[-1]
avg_latest_price = float(np.nanmean(latest_prices.values))
total_volume = float(df["Volume"].dropna().sum())

# Total return %
total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

# Volatility (annualized)
vol_ann = returns.std() * np.sqrt(252) * 100

# Sharpe (annualized)
excess_daily = returns.sub(rf_daily)
sharpe_ann = (excess_daily.mean() / returns.std()) * np.sqrt(252)
sharpe_ann = sharpe_ann.replace([np.inf, -np.inf], np.nan)

# Max drawdown
cum = (1 + returns).cumprod()
dd = (cum / cum.cummax() - 1) * 100
max_dd = dd.min()

# Beta/Alpha vs benchmark (if benchmark exists)
beta = pd.Series(index=prices.columns, dtype=float)
alpha_ann = pd.Series(index=prices.columns, dtype=float)
excess_vs_bm = pd.Series(index=prices.columns, dtype=float)

if benchmark_name in returns.columns:
    bm = returns[benchmark_name].dropna()
    var_bm = float(np.nanvar(bm.values, ddof=1)) if bm.shape[0] > 2 else np.nan

    for col in returns.columns:
        r = returns[col].dropna()
        aligned = pd.concat([r, bm], axis=1, join="inner").dropna()
        if aligned.shape[0] < 5 or not np.isfinite(var_bm) or var_bm == 0:
            beta[col] = np.nan
            alpha_ann[col] = np.nan
            excess_vs_bm[col] = np.nan
            continue

        cov = float(np.cov(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values, ddof=1)[0, 1])
        b = cov / var_bm
        beta[col] = b

        # alpha (annualized): mean(r - (rf + beta*(bm - rf))) * 252
        r_excess = aligned.iloc[:, 0] - rf_daily
        bm_excess = aligned.iloc[:, 1] - rf_daily
        alpha_daily = float(np.nanmean((r_excess - b * bm_excess).values))
        alpha_ann[col] = alpha_daily * 252.0

        # excess return vs benchmark (total return diff)
        excess_vs_bm[col] = float(total_return[col] - total_return[benchmark_name])

# VaR / CVaR (parametric-free using historical, daily)
def var_cvar(series: pd.Series, level=0.95):
    x = series.dropna().values
    if x.size < 10:
        return np.nan, np.nan
    q = np.quantile(x, 1 - level)  # e.g., 5% quantile
    cvar = x[x <= q].mean() if np.any(x <= q) else np.nan
    return q, cvar

var95 = pd.Series(index=returns.columns, dtype=float)
cvar95 = pd.Series(index=returns.columns, dtype=float)
for col in returns.columns:
    v, cv = var_cvar(returns[col], 0.95)
    var95[col] = v * 100
    cvar95[col] = cv * 100

summary = pd.DataFrame({
    "Latest Price": latest_prices.round(2),
    "Total Return %": total_return.round(2),
    "Volatility % (ann)": vol_ann.round(2),
    "Sharpe (ann)": sharpe_ann.round(2),
    "Max Drawdown %": max_dd.round(2),
    "VaR 95% (daily)": var95.round(2),
    "CVaR 95% (daily)": cvar95.round(2),
})

if benchmark_name in returns.columns:
    summary["Beta vs TASI"] = beta.round(2)
    summary["Alpha (ann)"] = alpha_ann.round(2)
    summary["Excess Return vs TASI %"] = excess_vs_bm.round(2)

summary = summary.sort_values("Total Return %", ascending=False)

normalized = prices.div(prices.iloc[0]).mul(100)

# -----------------------------
# Top KPIs (cards)
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Assets Selected</div><div class="kpi-value">{len(selected_assets)}</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Average Latest Price (SAR)</div><div class="kpi-value">{avg_latest_price:,.2f}</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Volume</div><div class="kpi-value">{int(total_volume):,}</div></div>""", unsafe_allow_html=True)
c4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Period</div><div class="kpi-value">{prices.index.min().date()} → {prices.index.max().date()}</div></div>""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="margin-top:10px;">
      <span class="pill">Risk-free: {risk_free:.2f}%</span>
      <span class="pill">Rolling window: {roll_window}d</span>
      <span class="pill">Last refresh: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_perf, tab_risk, tab_port, tab_data = st.tabs(
    ["Overview", "Performance", "Risk", "Portfolio", "Data"]
)

# -------- Overview --------
with tab_overview:
    left, right = st.columns([2, 1])

    with left:
        fig_price = px.line(
            df.dropna(subset=["Close"]),
            x="Date",
            y="Close",
            color="Asset",
            title="Price Trend",
            labels={"Close": "Closing Price (SAR)"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with right:
        rank = latest_prices.dropna().sort_values(ascending=False).reset_index()
        rank.columns = ["Asset", "Latest Price"]
        fig_rank = px.bar(rank, x="Latest Price", y="Asset", orientation="h", title="Latest Price Ranking")
        st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("Quick Snapshot")
    st.dataframe(summary, use_container_width=True, height=280)

# -------- Performance --------
with tab_perf:
    fig_norm = px.line(normalized, title="Normalized Performance (Base = 100)")
    st.plotly_chart(fig_norm, use_container_width=True)

    # Rolling Sharpe (approx)
    st.subheader("Rolling Metrics")
    cols = st.columns(2)

    with cols[0]:
        roll_vol = returns.rolling(roll_window).std() * np.sqrt(252) * 100
        fig_rv = px.line(roll_vol, title=f"Rolling Volatility (ann, {roll_window}d)")
        st.plotly_chart(fig_rv, use_container_width=True)

    with cols[1]:
        roll_sh = (returns.sub(rf_daily).rolling(roll_window).mean() / returns.rolling(roll_window).std()) * np.sqrt(252)
        fig_rs = px.line(roll_sh, title=f"Rolling Sharpe (ann, {roll_window}d)")
        st.plotly_chart(fig_rs, use_container_width=True)

# -------- Risk --------
with tab_risk:
    c1, c2 = st.columns(2)
    with c1:
        fig_vol = px.bar(summary.reset_index().rename(columns={"index": "Asset"}), x="Asset", y="Volatility % (ann)", title="Annualized Volatility (Risk)")
        st.plotly_chart(fig_vol, use_container_width=True)
    with c2:
        fig_dd = px.bar(summary.reset_index().rename(columns={"index": "Asset"}), x="Asset", y="Max Drawdown %", title="Maximum Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Diversification View (Correlation)")
    corr = returns.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap (Daily Returns)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Tail Risk (Daily)")
    tail = summary[["VaR 95% (daily)", "CVaR 95% (daily)"]].copy()
    fig_tail = px.bar(
        tail.reset_index().rename(columns={"index": "Asset"}),
        x="Asset",
        y=["VaR 95% (daily)", "CVaR 95% (daily)"],
        barmode="group",
        title="VaR vs CVaR (95%) — downside risk estimate"
    )
    st.plotly_chart(fig_tail, use_container_width=True)

# -------- Portfolio --------
with tab_port:
    st.write("Build and compare portfolios (random optimizer). This demonstrates risk/return understanding beyond single-asset charts.")

    # Use returns columns only with enough data
    usable_assets = [c for c in returns.columns if returns[c].dropna().shape[0] >= 10]
    ret_mat = returns[usable_assets].dropna()

    if ret_mat.shape[0] < 15 or len(usable_assets) < 2:
        st.warning("Not enough clean data for portfolio simulation. Try a longer range or select more assets.")
    else:
        n = len(usable_assets)
        mean_daily = ret_mat.mean()
        cov_daily = ret_mat.cov()

        sims = st.slider("Number of simulated portfolios", 200, 5000, 1500, 100)
        rng = np.random.default_rng(42)

        W = rng.random((sims, n))
        W = W / W.sum(axis=1, keepdims=True)

        # Portfolio daily mean/std
        port_mean = W @ mean_daily.values
        port_var = np.einsum("ij,jk,ik->i", W, cov_daily.values, W)
        port_std = np.sqrt(port_var)

        port_ret_ann = port_mean * 252 * 100
        port_vol_ann = port_std * np.sqrt(252) * 100
        port_sharpe = ((port_mean - rf_daily) / port_std) * np.sqrt(252)

        best_idx = np.nanargmax(port_sharpe)
        minvol_idx = np.nanargmin(port_vol_ann)

        best_w = W[best_idx]
        minvol_w = W[minvol_idx]

        frontier_df = pd.DataFrame({
            "Return % (ann)": port_ret_ann,
            "Volatility % (ann)": port_vol_ann,
            "Sharpe (ann)": port_sharpe,
        })

        fig_frontier = px.scatter(
            frontier_df,
            x="Volatility % (ann)",
            y="Return % (ann)",
            color="Sharpe (ann)",
            title="Efficient Frontier (Random Simulation)",
        )
        # Highlight best Sharpe and min-vol
        fig_frontier.add_trace(go.Scatter(
            x=[port_vol_ann[best_idx]], y=[port_ret_ann[best_idx]],
            mode="markers", name="Max Sharpe",
            marker=dict(size=14, symbol="star")
        ))
        fig_frontier.add_trace(go.Scatter(
            x=[port_vol_ann[minvol_idx]], y=[port_ret_ann[minvol_idx]],
            mode="markers", name="Min Vol",
            marker=dict(size=12, symbol="diamond")
        ))
        st.plotly_chart(fig_frontier, use_container_width=True)

        def weights_table(weights, label):
            out = pd.DataFrame({"Asset": usable_assets, "Weight": weights})
            out["Weight"] = (out["Weight"] * 100).round(2)
            out = out.sort_values("Weight", ascending=False).reset_index(drop=True)
            st.subheader(label)
            st.dataframe(out, use_container_width=True, height=220)
            return out

        a, b = st.columns(2)
        with a:
            best_tbl = weights_table(best_w, "Max Sharpe Portfolio (weights %)")
        with b:
            minvol_tbl = weights_table(minvol_w, "Min Volatility Portfolio (weights %)")

        # Compare growth of $1
        st.subheader("Growth Comparison (Base = 1.0)")
        base = (1 + ret_mat).cumprod()

        # Equal-weight
        ew = np.repeat(1 / n, n)
        ew_series = (1 + (ret_mat @ ew)).cumprod()

        best_series = (1 + (ret_mat @ best_w)).cumprod()
        minvol_series = (1 + (ret_mat @ minvol_w)).cumprod()

        comp = pd.DataFrame({
            "Equal-weight": ew_series,
            "Max Sharpe": best_series,
            "Min Vol": minvol_series,
        }, index=ret_mat.index)

        if benchmark_name in comp.columns:
            pass

        fig_comp = px.line(comp, title="Portfolio Growth (Base=1)")
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown('<p class="small-note">Note: Optimizer uses random simulation (robust for assessments, avoids heavy dependencies). Results vary slightly with data range.</p>', unsafe_allow_html=True)

# -------- Data --------
with tab_data:
    st.subheader("Data Quality")
    coverage = pd.DataFrame({
        "Asset": prices.columns,
        "Non-null price points": [int(prices[c].notna().sum()) for c in prices.columns],
        "Missing price points": [int(prices[c].isna().sum()) for c in prices.columns],
    }).sort_values("Non-null price points", ascending=False)
    st.dataframe(coverage, use_container_width=True, height=200)

    st.subheader("Raw Data (Sample)")
    st.dataframe(df.head(800), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "market_data.csv", "text/csv")

# -----------------------------
# Auto Insights (Storytelling)
# -----------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("Key Insights (Auto-generated)")

best_asset = summary["Total Return %"].idxmax()
worst_asset = summary["Total Return %"].idxmin()
highest_vol = summary["Volatility % (ann)"].idxmax()
best_sharpe = summary["Sharpe (ann)"].idxmax()

st.write(f"✅ **Best performer (Total Return):** {best_asset} ({summary.loc[best_asset,'Total Return %']}%).")
st.write(f"⚠️ **Worst performer (Total Return):** {worst_asset} ({summary.loc[worst_asset,'Total Return %']}%).")
st.write(f"📌 **Highest volatility (Risk):** {highest_vol} ({summary.loc[highest_vol,'Volatility % (ann)']}%).")
st.write(f"🏅 **Best risk-adjusted return (Sharpe):** {best_sharpe} ({summary.loc[best_sharpe,'Sharpe (ann)']}).")

if benchmark_name in summary.columns or benchmark_name in returns.columns:
    if benchmark_name in returns.columns:
        ex = summary.get("Excess Return vs TASI %", pd.Series(dtype=float))
        if not ex.empty and ex.dropna().shape[0] > 0:
            top_ex = ex.dropna().idxmax()
            st.write(f"📈 **Top outperformance vs TASI:** {top_ex} ({ex.loc[top_ex]}% excess return).")

    if "Beta vs TASI" in summary.columns:
        bmax = summary["Beta vs TASI"].dropna()
        if bmax.shape[0] > 0:
            most_sensitive = bmax.idxmax()
            st.write(f"🧠 **Market sensitivity:** {most_sensitive} has the highest Beta vs TASI ({summary.loc[most_sensitive,'Beta vs TASI']}).")

st.markdown(
    '<p class="small-note">Volatility reflects price fluctuation (risk). Sharpe compares return per unit of risk. Correlation helps assess diversification. Beta/Alpha are measured relative to the benchmark (TASI).</p>',
    unsafe_allow_html=True
)
st.markdown('<p class="small-note"><b>Disclaimer:</b> Educational dashboard for assessment purposes only (not investment advice).</p>', unsafe_allow_html=True)
