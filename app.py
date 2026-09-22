import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from pathlib import Path

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ENTSO-e Load Analysis",
    page_icon="⚡",
    layout="wide",
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# ------------------------------------------------------------------
# Data loading (cached so the CSV is read only once)
# ------------------------------------------------------------------
DATA_PATH = Path("monthly_hourly_load_values_2024.csv")

# Friendly names for ENTSO-e country codes
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

@st.cache_data(show_spinner="Loading ENTSO-e data…")
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=False)
def prepare_country(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Replicates the notebook's preparation for a single country."""
    c = df[df["CountryCode"] == country_code].copy()
    if c.empty:
        return c

    c = c.filter(items=["DateUTC", "Value"])

    # Parse datetime once
    c["_dt"] = pd.to_datetime(c["DateUTC"], format="mixed", dayfirst=True)

    c["Date"]  = c["_dt"].dt.date
    c["Time"]  = c["_dt"].dt.time
    c["Month"] = c["_dt"].dt.month_name()
    c["Year"]  = c["_dt"].dt.year
    c["Day"]   = c["_dt"].dt.day_name()

    c = c.drop(columns=["_dt"])
    c = c.set_index(pd.to_datetime(c["DateUTC"], format="mixed", dayfirst=True))
    c = c.drop(columns=["DateUTC"])
    return c

# ------------------------------------------------------------------
# Sidebar – country selector
# ------------------------------------------------------------------
st.sidebar.title("⚡ ENTSO-e Dashboard")
st.sidebar.markdown("Hourly Load Values (2024)")

if not DATA_PATH.exists():
    st.error(f"Data file not found: `{DATA_PATH.resolve()}`")
    st.stop()

raw = load_raw(str(DATA_PATH))

# Only show country codes that actually exist in the file
available_codes = sorted(raw["CountryCode"].dropna().unique().tolist())

# Build display labels: "NL — Netherlands"
label_map = {
    code: f"{code} — {COUNTRY_NAMES.get(code, code)}"
    for code in available_codes
}

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
# Prepare data for the chosen country
# ------------------------------------------------------------------
df = prepare_country(raw, country)

if df.empty:
    st.warning(f"No data found for country code `{country}`.")
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title(f"⚡ Energy Load Analysis — {country_display}")
st.caption("Hourly load values, ENTSO-e 2024")

# ------------------------------------------------------------------
# 1. Overview KPIs
# ------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows",        f"{len(df):,}")
c2.metric("Mean (MW)",   f"{df['Value'].mean():,.1f}")
c3.metric("Peak (MW)",   f"{df['Value'].max():,.1f}")
c4.metric("Min (MW)",    f"{df['Value'].min():,.1f}")

# ------------------------------------------------------------------
# 2. Data preview / info / describe / nulls
# ------------------------------------------------------------------
with st.expander("📋 Dataset overview (head / info / describe / nulls)", expanded=False):
    st.markdown("**First 5 rows**")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("**Describe**")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("**Null values**")
    st.dataframe(df.isnull().sum().rename("nulls"), use_container_width=True)

    st.markdown("**Years present**")
    st.write(df["Year"].unique().tolist(), "— total:", df["Year"].nunique())

# ------------------------------------------------------------------
# 3. Monthly aggregation
# ------------------------------------------------------------------
st.subheader("📊 Monthly consumption overview")

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

monthly = (
    df.groupby("Month")["Value"]
      .agg(["max", "mean", "min"])
      .reindex(month_order)
      .dropna(how="all")
)
st.dataframe(monthly.style.format("{:,.1f}"), use_container_width=True)

# ------------------------------------------------------------------
# 4. Distribution
# ------------------------------------------------------------------
st.subheader("📈 Distribution of hourly load")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["Value"], kde=True, ax=ax, color="#1f77b4")
ax.set_xlabel("Load (MW)")
ax.set_ylabel("Frequency")
ax.set_title("Energy distribution")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ------------------------------------------------------------------
# 5. Monthly line plot (average per month)
# ------------------------------------------------------------------
st.subheader("📉 Average load per month")
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(
    data=df, x="Month", y="Value",
    order=month_order, estimator="mean", errorbar=None,
    marker="o", ax=ax, color="#d62728",
)
ax.set_xlabel("Month")
ax.set_ylabel("Load (MW)")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ------------------------------------------------------------------
# 6. Boxplot per month
# ------------------------------------------------------------------
st.subheader("📦 Load distribution per month")
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x=df.index.month, y=df["Value"], ax=ax, color="#4c72b0")
ax.set_xlabel("Month")
ax.set_ylabel("Load (MW)")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ------------------------------------------------------------------
# 7. Jan / Jun / Dec comparison
# ------------------------------------------------------------------
st.subheader("🔍 January vs June vs December")

months_map = {1: "January", 6: "June", 12: "December"}
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

for ax, (m_num, m_name) in zip(axes, months_map.items()):
    sub = df[df.index.month == m_num]
    ax.plot(sub.index, sub["Value"], linewidth=1.2, label=m_name)
    ax.set_title(f"{m_name} — hourly load")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Datetime")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ------------------------------------------------------------------
# 8. All 12 months (unified y-axis)
# ------------------------------------------------------------------
st.subheader("🗓️ Hourly load for every month (unified Y-axis)")
fig, axes = plt.subplots(12, 1, figsize=(20, 26), sharey=True)
fig.suptitle(f"Energy consumption per month — {country_display}", fontsize=16)

y_min, y_max = df["Value"].min(), df["Value"].max()
for i, ax in enumerate(axes):
    m_num = i + 1
    m_data = df[df.index.month == m_num]
    if m_data.empty:
        ax.set_visible(False)
        continue
    ax.plot(m_data.index, m_data["Value"], linewidth=1.4,
            label=month_order[i], color="steelblue")
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("MW", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

for ax in axes[8:]:
    ax.set_xlabel("Date", fontsize=10)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption("Built with Streamlit · Data source: ENTSO-e Transparency Platform")
