import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="ENTSO-e Load Analysis", page_icon="⚡", layout="wide")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titlesize"] = 12

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DATA_PATH = Path("monthly_hourly_load_values_2024.csv")

COUNTRY_NAMES = {
    "NL": "Netherlands", "DE": "Germany", "BE": "Belgium",
    "FR": "France", "ES": "Spain", "IT": "Italy",
    "AT": "Austria", "PL": "Poland", "DK": "Denmark",
    "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "PT": "Portugal", "CH": "Switzerland", "CZ": "Czechia",
    "IE": "Ireland", "GR": "Greece", "HU": "Hungary",
    "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
    "SK": "Slovakia", "SI": "Slovenia", "LT": "Lithuania",
    "LV": "Latvia", "EE": "Estonia", "LU": "Luxembourg",
}

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# ------------------------------------------------------------------
# Safe plotting helper — one bad plot won't kill the app
# ------------------------------------------------------------------
def safe_plot(title: str, plot_fn, *args, **kwargs):
    """Wrap any plotting function in try/except and render it."""
    try:
        fig = plot_fn(*args, **kwargs)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"⚠️ Could not render **{title}**: `{type(e).__name__}: {e}`")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading ENTSO-e data…")
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def prepare_country(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Replicates the notebook's preparation for a single country."""
    c = df[df["CountryCode"] == country_code].copy()
    if c.empty:
        return c

    c = c.filter(items=["DateUTC", "Value"])
    c["_dt"] = pd.to_datetime(c["DateUTC"], format="mixed", dayfirst=True)

    # Ordered categoricals so plots always show the right order
    c["Month"] = pd.Categorical(c["_dt"].dt.month_name(),
                                categories=MONTH_ORDER, ordered=True)
    c["Day"]   = pd.Categorical(c["_dt"].dt.day_name(),
                                categories=DAY_ORDER, ordered=True)
    c["Date"]  = c["_dt"].dt.date
    c["Time"]  = c["_dt"].dt.time
    c["Year"]  = c["_dt"].dt.year
    c["Hour"]  = c["_dt"].dt.hour
    c["MonthNum"] = c["_dt"].dt.month

    c = c.set_index("_dt")
    c.index.name = "Datetime"
    c = c.drop(columns=["DateUTC"])
    return c


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("⚡ ENTSO-e Dashboard")
st.sidebar.markdown("Hourly Load Values — 2024")

if not DATA_PATH.exists():
    st.error(f"Data file not found: `{DATA_PATH.resolve()}`")
    st.stop()

raw = load_raw(str(DATA_PATH))
available_codes = sorted(raw["CountryCode"].dropna().unique().tolist())

label_map = {code: f"{code} — {COUNTRY_NAMES.get(code, code)}"
             for code in available_codes}

selected_label = st.sidebar.selectbox(
    "🌍 Select a country",
    options=[label_map[c] for c in available_codes],
    index=available_codes.index("NL") if "NL" in available_codes else 0,
)
country = selected_label.split(" — ")[0]
country_display = label_map[country]

st.sidebar.success(f"Active: **{country_display}**")
st.sidebar.caption(f"Rows in file: {len(raw):,}")

# ------------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------------
df = prepare_country(raw, country)
if df.empty:
    st.warning(f"No data found for country code `{country}`.")
    st.stop()

# ------------------------------------------------------------------
# Header + KPIs
# ------------------------------------------------------------------
st.title(f"⚡ Energy Load Analysis — {country_display}")
st.caption("Hourly load values · ENTSO-e Transparency Platform · 2024")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",       f"{len(df):,}")
c2.metric("Mean (MW)",  f"{df['Value'].mean():,.1f}")
c3.metric("Median (MW)",f"{df['Value'].median():,.1f}")
c4.metric("Peak (MW)",  f"{df['Value'].max():,.1f}")
c5.metric("Min (MW)",   f"{df['Value'].min():,.1f}")

# ------------------------------------------------------------------
# Tabs — organized like the notebook
# ------------------------------------------------------------------
tab_overview, tab_monthly, tab_dist, tab_hourly, tab_heatmap, tab_detail = st.tabs(
    ["📋 Overview", "📊 Monthly", "📈 Distribution",
     "🕐 Daily/Hourly", "🔥 Heatmaps", "🔍 Details"]
)

# ==================================================================
# TAB 1 — Overview
# ==================================================================
with tab_overview:
    st.subheader("Dataset overview")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**First 5 rows**")
        st.dataframe(df.head(), use_container_width=True)
    with col_b:
        st.markdown("**Describe**")
        st.dataframe(df.describe(), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Null values**")
        st.dataframe(df.isnull().sum().rename("nulls"), use_container_width=True)
    with col_d:
        st.markdown("**Years present**")
        st.write(df["Year"].unique().tolist(), "— total:", df["Year"].nunique())

# ==================================================================
# TAB 2 — Monthly analysis
# ==================================================================
with tab_monthly:
    st.subheader("📊 Monthly consumption overview")
    monthly = (df.groupby("Month", observed=True)["Value"]
                 .agg(["max", "mean", "min"])
                 .reindex(MONTH_ORDER)
                 .dropna(how="all"))
    st.dataframe(monthly.style.format("{:,.1f}"), use_container_width=True)

    # --- Average per month (line) ---
    st.subheader("📉 Average load per month")
    def _avg_month():
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=df, x="Month", y="Value",
                     estimator="mean", errorbar=None,
                     marker="o", ax=ax, color="#d62728")
        ax.set_xlabel("Month"); ax.set_ylabel("Load (MW)")
        ax.tick_params(axis="x", rotation=45)
        return fig
    safe_plot("Average load per month", _avg_month)

    # --- Boxplot per month ---
    st.subheader("📦 Load distribution per month")
    def _box_month():
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x="Month", y="Value",
                    order=MONTH_ORDER, ax=ax, color="#4c72b0")
        ax.set_xlabel("Month"); ax.set_ylabel("Load (MW)")
        ax.tick_params(axis="x", rotation=45)
        return fig
    safe_plot("Boxplot per month", _box_month)

    # --- Jan / Jun / Dec comparison ---
    st.subheader("🔍 January vs June vs December")
    def _jan_jun_dec():
        months_map = {1: "January", 6: "June", 12: "December"}
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        for ax, (m_num, m_name) in zip(axes, months_map.items()):
            sub = df[df.index.month == m_num]
            ax.plot(sub.index, sub["Value"], linewidth=1.2, label=m_name)
            ax.set_title(f"{m_name} — hourly load")
            ax.set_ylabel("MW"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
        axes[-1].set_xlabel("Datetime")
        plt.tight_layout()
        return fig
    safe_plot("Jan/Jun/Dec comparison", _jan_jun_dec)

# ==================================================================
# TAB 3 — Distribution
# ==================================================================
with tab_dist:
    st.subheader("📈 Distribution of hourly load")
    def _hist():
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["Value"], kde=True, ax=ax, color="#1f77b4")
        ax.set_xlabel("Load (MW)"); ax.set_ylabel("Frequency")
        ax.set_title("Energy distribution")
        return fig
    safe_plot("Histogram of hourly load", _hist)

    st.subheader("📈 Distribution by month (violin)")
    def _violin():
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.violinplot(data=df, x="Month", y="Value",
                       order=MONTH_ORDER, ax=ax, palette="viridis", inner="quart")
        ax.set_xlabel("Month"); ax.set_ylabel("Load (MW)")
        ax.tick_params(axis="x", rotation=45)
        return fig
    safe_plot("Violin by month", _violin)

# ==================================================================
# TAB 4 — Daily / Hourly patterns
# ==================================================================
with tab_hourly:
    st.subheader("🕐 Average load by hour of day")
    def _hour_curve():
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=df, x="Hour", y="Value",
                     estimator="mean", errorbar=None,
                     marker="o", ax=ax, color="#2ca02c")
        ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Load (MW)")
        ax.set_xticks(range(0, 24))
        return fig
    safe_plot("Average load by hour", _hour_curve)

    st.subheader("📅 Average load by day of week")
    def _day_curve():
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df, x="Day", y="Value",
                    order=DAY_ORDER, ax=ax, color="#9467bd")
        ax.set_xlabel("Day"); ax.set_ylabel("Mean load (MW)")
        ax.tick_params(axis="x", rotation=45)
        return fig
    safe_plot("Average load by weekday", _day_curve)

    st.subheader("🕐 Load profile: weekday vs weekend by hour")
    def _wd_vs_we():
        tmp = df.copy()
        tmp["is_weekend"] = tmp["Day"].isin(["Saturday", "Sunday"])
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=tmp, x="Hour", y="Value", hue="is_weekend",
                     estimator="mean", errorbar=None, marker="o", ax=ax)
        ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Mean load (MW)")
        ax.set_xticks(range(0, 24))
        ax.legend(title="Weekend?", labels=["Weekday", "Weekend"])
        return fig
    safe_plot("Weekday vs weekend hourly profile", _wd_vs_we)

# ==================================================================
# TAB 5 — Heatmaps
# ==================================================================
with tab_heatmap:
    st.subheader("🔥 Load heatmap — Month × Hour of day")
    def _heat_month_hour():
        pivot = (df.groupby(["MonthNum", "Hour"], observed=True)["Value"]
                   .mean().unstack())
        pivot = pivot.reindex(range(1, 13))
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Mean load (MW)"})
        ax.set_yticklabels(MONTH_ORDER, rotation=0)
        ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Month")
        return fig
    safe_plot("Month × Hour heatmap", _heat_month_hour)

    st.subheader("🔥 Load heatmap — Day of week × Hour of day")
    def _heat_day_hour():
        pivot = (df.groupby(["Day", "Hour"], observed=True)["Value"]
                   .mean().unstack().reindex(DAY_ORDER))
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(pivot, cmap="viridis", ax=ax,
                    cbar_kws={"label": "Mean load (MW)"})
        ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Day")
        return fig
    safe_plot("Day × Hour heatmap", _heat_day_hour)

# ==================================================================
# TAB 6 — Detail (all 12 months, one at a time)
# ==================================================================
with tab_detail:
    st.subheader("🗓️ Explore any month in detail")
    month_choice = st.selectbox("Pick a month", MONTH_ORDER, index=0)
    m_num = MONTH_ORDER.index(month_choice) + 1
    m_data = df[df.index.month == m_num]

    if m_data.empty:
        st.info(f"No data for {month_choice}.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Mean",  f"{m_data['Value'].mean():,.1f} MW")
        d2.metric("Peak",  f"{m_data['Value'].max():,.1f} MW")
        d3.metric("Min",   f"{m_data['Value'].min():,.1f} MW")

        def _one_month():
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(m_data.index, m_data["Value"],
                    linewidth=1.4, color="steelblue")
            ax.set_title(f"{month_choice} — hourly load")
            ax.set_xlabel("Date"); ax.set_ylabel("MW"); ax.grid(alpha=0.3)
            plt.tight_layout()
            return fig
        safe_plot(f"{month_choice} hourly load", _one_month)

        st.markdown("**Daily mean within the month**")
        def _daily_mean():
            daily = m_data.groupby(m_data.index.date)["Value"].mean()
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(daily.index, daily.values, marker="o",
                    linewidth=1.4, color="#d62728")
            ax.set_xlabel("Date"); ax.set_ylabel("Daily mean (MW)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            return fig
        safe_plot("Daily mean", _daily_mean)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption("Built with Streamlit · Data source: ENTSO-e Transparency Platform")
