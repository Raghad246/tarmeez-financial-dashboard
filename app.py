import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Saudi Market Snapshot", layout="wide")

st.title("📈 Saudi Market Snapshot (TASI) — Live Financial Dashboard")
st.caption("Interactive dashboard built for Tarmeez Capital practical assessment.")

# ---- Demo public dataset (you will replace later with your chosen public data source) ----
@st.cache_data
def load_demo_data():
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    sectors = ["Banks", "Energy", "Materials", "Retail", "Telecom", "Healthcare"]
    rows = []
    rng = np.random.default_rng(42)

    for s in sectors:
        base = 100 + rng.normal(0, 1)
        prices = base + np.cumsum(rng.normal(0, 0.8, len(dates)))
        volume = rng.integers(50_000, 300_000, len(dates))
        rows.append(pd.DataFrame({
            "date": dates,
            "sector": s,
            "index_value": prices,
            "volume": volume
        }))
    return pd.concat(rows, ignore_index=True)

df = load_demo_data()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
sector = st.sidebar.multiselect("Sector", sorted(df["sector"].unique()), default=sorted(df["sector"].unique()))
start, end = st.sidebar.date_input("Date range", value=(df["date"].min().date(), df["date"].max().date()))

mask = (
    df["sector"].isin(sector) &
    (df["date"].dt.date >= start) &
    (df["date"].dt.date <= end)
)
dff = df.loc[mask].copy()

# ---- KPI calculations ----
latest = dff.sort_values("date").groupby("sector").tail(1)
kpi_cols = st.columns(4)
kpi_cols[0].metric("Sectors Selected", len(sector))
kpi_cols[1].metric("Date Range", f"{start} → {end}")
kpi_cols[2].metric("Latest Avg Index", f"{latest['index_value'].mean():.2f}")
kpi_cols[3].metric("Latest Total Volume", f"{int(latest['volume'].sum()):,}")

st.divider()

# ---- Charts ----
c1, c2 = st.columns([2, 1])

with c1:
    fig = px.line(
        dff,
        x="date",
        y="index_value",
        color="sector",
        title="Sector Index Trend",
        labels={"index_value": "Index Value", "date": "Date"}
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    latest_sorted = latest.sort_values("index_value", ascending=False)
    fig2 = px.bar(
        latest_sorted,
        x="index_value",
        y="sector",
        orientation="h",
        title="Latest Sector Ranking",
        labels={"index_value": "Latest Index Value", "sector": "Sector"}
    )
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    vol = dff.groupby("sector")["volume"].sum().reset_index().sort_values("volume", ascending=False)
    fig3 = px.bar(vol, x="sector", y="volume", title="Total Volume by Sector")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    dff["daily_return"] = dff.groupby("sector")["index_value"].pct_change()
    volat = dff.groupby("sector")["daily_return"].std().reset_index().sort_values("daily_return", ascending=False)
    fig4 = px.bar(volat, x="sector", y="daily_return", title="Volatility (Std of Daily Returns)")
    st.plotly_chart(fig4, use_container_width=True)

st.info("✅ This is a starter dashboard. Next step: replace the demo data with a public TASI/finance dataset and document the source in README.")

