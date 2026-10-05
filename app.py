"""
app.py - Automated Data Cleaning and Reporting Dashboard
Run with:  streamlit run app.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import reporting as rp
from cleaning import clean_data_with_report, find_column, generate_sample_data

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
REPORT_DIR = os.path.join(BASE, "reports")
RAW_PATH = os.path.join(DATA_DIR, "raw_data.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

st.set_page_config(page_title="Data Cleaning & Reporting", page_icon="🧹", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.title {font-size: 2rem; font-weight: 700; color: #0F766E; margin-bottom: 0;}
.subtitle {color: #64748b; margin-top: 0;}
.kpi {background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #0F766E;
      border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.kpi .label {font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {font-size: 1.6rem; font-weight: 700; color: #0f172a;}
.kpi.accent {border-left-color: #F59E0B;}
.insight {background: #f0fdfa; border-left: 4px solid #14B8A6; padding: 8px 12px;
          border-radius: 6px; margin-bottom: 6px;}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, accent=False):
    col.markdown(f"<div class='kpi {'accent' if accent else ''}'><div class='label'>{label}</div>"
                 f"<div class='value'>{value}</div></div>", unsafe_allow_html=True)


def show(fig, msg="Not enough data or required columns are missing for this chart."):
    if fig is None:
        st.info(msg)
    else:
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_default():
    if not os.path.exists(RAW_PATH):
        generate_sample_data().to_csv(RAW_PATH, index=False)
    return pd.read_csv(RAW_PATH)


def read_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


@st.cache_data
def run_cleaning(raw):
    return clean_data_with_report(raw)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧹 Controls")
uploaded = st.sidebar.file_uploader("Upload a dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

try:
    if uploaded is not None:
        raw = read_upload(uploaded)
        source = uploaded.name
    else:
        raw = load_default()
        source = "Sample dataset (data/raw_data.csv)"
except Exception as e:
    st.error(f"Could not read the file. Please upload a valid CSV or Excel file.\n\nDetails: {e}")
    st.stop()

if raw.empty or raw.shape[1] == 0:
    st.error("The uploaded file has no data. Please upload a file with at least one row and one column.")
    st.stop()

try:
    cleaned, report = run_cleaning(raw)
except Exception as e:
    st.error(f"Cleaning failed for this dataset: {e}")
    st.stop()

# Save cleaned output
try:
    if uploaded is None:
        cleaned.to_csv(CLEAN_PATH, index=False)
    else:
        cleaned.to_csv(os.path.join(REPORT_DIR, f"cleaned_{os.path.splitext(uploaded.name)[0]}.csv"), index=False)
except OSError:
    pass

st.sidebar.caption(f"Source: **{source}**")
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

filtered = cleaned.copy()
date_col = find_column(cleaned, "purchase_date", "date")
if date_col and pd.api.types.is_datetime64_any_dtype(cleaned[date_col]) and cleaned[date_col].notna().any():
    dmin, dmax = cleaned[date_col].min().date(), cleaned[date_col].max().date()
    rng = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        filtered = filtered[(filtered[date_col].dt.date >= rng[0]) & (filtered[date_col].dt.date <= rng[1])]
else:
    st.sidebar.caption("No date column found - date filter disabled.")

for label, key in [("Category", "category"), ("Region", "region"), ("Product", "product")]:
    col = find_column(cleaned, key)
    if col:
        options = sorted(cleaned[col].astype(str).unique())
        chosen = st.sidebar.multiselect(label, options, placeholder=f"All {label.lower()}s")
        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]
    else:
        st.sidebar.caption(f"No '{key}' column - filter disabled.")

# ---------------------------------------------------------------- header + KPIs
st.markdown("<p class='title'>Automated Data Cleaning & Reporting Dashboard</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload raw data, clean it automatically, check quality and download reports.</p>",
            unsafe_allow_html=True)

b, a = report["before"], report["after"]
k = st.columns(6)
kpi(k[0], "Total Records", f"{b['rows']:,}")
kpi(k[1], "Clean Records", f"{a['rows']:,}")
kpi(k[2], "Missing Fixed", f"{b['missing']:,}", True)
kpi(k[3], "Duplicates Removed", f"{b['duplicates']:,}", True)
kpi(k[4], "Invalid Fixed", f"{b['invalid']:,}", True)
kpi(k[5], "Quality Score", f"{a['score']}%")
st.write("")

missing_cols = rp.missing_business_columns(cleaned)
if missing_cols:
    st.warning(f"These expected columns were not found: {', '.join(missing_cols)}. "
               "Cleaning still works, but some business charts will be skipped.")

tabs = st.tabs(["📋 Overview", "🧽 Data Cleaning", "✅ Data Quality", "📈 Business Analysis", "📄 Automated Report"])

# ---------------------------------------------------------------- 1. Overview
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Dataset Summary")
        st.dataframe(rp.quality_summary_table(report), hide_index=True, width="stretch")
        st.metric("Quality score", f"{a['score']}%", delta=f"{a['score'] - b['score']:+.1f} pts")
    with c2:
        st.subheader("Cleaning Summary")
        for op in report["operations"]:
            st.markdown(f"- {op}")
    st.subheader("Cleaned Data Preview")
    st.dataframe(filtered.head(100), width="stretch")

# ---------------------------------------------------------------- 2. Data Cleaning
with tabs[1]:
    st.subheader("Original (Raw) Dataset")
    st.dataframe(raw.head(100), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing Value Analysis")
        miss = pd.Series(report["missing_by_column"], name="Missing").to_frame()
        miss["% of rows"] = (100 * miss["Missing"] / b["rows"]).round(2)
        st.dataframe(miss.sort_values("Missing", ascending=False), width="stretch")
    with c2:
        st.subheader("Invalid Value Analysis")
        inv = report["invalid_by_column"]
        if inv:
            st.dataframe(pd.Series(inv, name="Invalid values").to_frame(), width="stretch")
        else:
            st.success("No invalid values found.")
        st.subheader("Duplicate Analysis")
        st.write(f"**{b['duplicates']}** duplicate records found and removed "
                 f"({100 * b['duplicates'] / b['rows']:.1f}% of raw rows).")

    st.subheader("Before vs After Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(rp.quality_summary_table(report), hide_index=True, width="stretch")
    with c2:
        show(rp.fig_before_after(report))

# ---------------------------------------------------------------- 3. Data Quality
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        show(rp.fig_missing_by_column(report), "No missing values in the raw data.")
        show(rp.fig_dtype_distribution(report))
    with c2:
        show(rp.fig_duplicates(report))
        show(rp.fig_outliers(raw), "No numeric columns to check for outliers.")
    show(rp.fig_completeness(report, cleaned))
    st.subheader("Column Profile")
    st.dataframe(rp.column_profile(raw, cleaned), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4. Business Analysis
with tabs[3]:
    if filtered.empty:
        st.warning("No rows match the selected filters.")
    else:
        rev = find_column(filtered, "revenue", "sales", "amount")
        if rev:
            m = st.columns(3)
            m[0].metric("Total Revenue", f"{filtered[rev].sum():,.2f}")
            m[1].metric("Average Order Value", f"{filtered[rev].mean():,.2f}")
            m[2].metric("Orders", f"{len(filtered):,}")
        c1, c2 = st.columns(2)
        with c1:
            show(rp.fig_revenue_by(filtered, "category"))
            show(rp.fig_top_products(filtered))
            show(rp.fig_quantity_sold(filtered))
        with c2:
            show(rp.fig_revenue_by(filtered, "region"))
            show(rp.fig_monthly_trend(filtered))
        show(rp.fig_customer_distribution(filtered))

# ---------------------------------------------------------------- 5. Automated Report
with tabs[4]:
    insights = rp.generate_insights(filtered, report)
    st.subheader("Automated Insights")
    for i in insights:
        st.markdown(f"<div class='insight'>{i}</div>", unsafe_allow_html=True)

    profile = rp.column_profile(raw, cleaned)
    quality_df = rp.quality_summary_table(report)

    st.subheader("Downloads")
    d = st.columns(3)
    d[0].download_button("⬇️ Raw dataset (CSV)", raw.to_csv(index=False), "raw_data.csv", "text/csv",
                         width="stretch")
    d[1].download_button("⬇️ Cleaned dataset (CSV)", cleaned.to_csv(index=False), "cleaned_data.csv",
                         "text/csv", width="stretch")
    d[2].download_button("⬇️ Cleaned dataset (Excel)", rp.df_to_excel_bytes(cleaned, "Cleaned Data"),
                         "cleaned_data.xlsx", width="stretch")
    d = st.columns(3)
    d[0].download_button("⬇️ Data quality report (CSV)", quality_df.to_csv(index=False),
                         "data_quality_report.csv", "text/csv", width="stretch")
    excel_report = rp.build_excel_report(report, insights, profile, cleaned)
    d[1].download_button("⬇️ Full report (Excel)", excel_report, "data_quality_report.xlsx",
                         width="stretch")

    figures = {
        "Before vs After": rp.fig_before_after(report),
        "Missing Values by Column": rp.fig_missing_by_column(report),
        "Revenue by Category": rp.fig_revenue_by(filtered, "category"),
        "Revenue by Region": rp.fig_revenue_by(filtered, "region"),
        "Top 10 Products": rp.fig_top_products(filtered),
        "Monthly Revenue Trend": rp.fig_monthly_trend(filtered),
    }
    html = rp.build_html_report(report, insights, figures, profile)
    d[2].download_button("⬇️ Summary report (HTML)", html, "summary_report.html", "text/html",
                         width="stretch")

    # Keep a copy in the reports/ folder
    try:
        with open(os.path.join(REPORT_DIR, "summary_report.html"), "w", encoding="utf-8") as f:
            f.write(html)
        with open(os.path.join(REPORT_DIR, "data_quality_report.xlsx"), "wb") as f:
            f.write(excel_report)
        st.caption("Copies of the reports were also saved in the reports/ folder.")
    except OSError:
        pass

st.markdown("---")
st.caption("Automated Data Cleaning & Reporting Dashboard · Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit")
