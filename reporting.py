"""
reporting.py
------------
Data quality tables, charts, automated insights and downloadable reports.
Every business chart checks that its columns exist and returns None if they
don't, so the dashboard never crashes on unusual datasets.
"""

import base64
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from cleaning import find_column, prepare  # noqa: E402

sns.set_theme(style="whitegrid", font="sans-serif")
PRIMARY = "#0F766E"
ACCENT = "#F59E0B"
PALETTE = ["#0F766E", "#14B8A6", "#F59E0B", "#64748B", "#0EA5E9", "#E11D48", "#84CC16"]


def _fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


# --------------------------------------------------------------------------
# Data quality tables
# --------------------------------------------------------------------------
def quality_summary_table(report):
    b, a = report["before"], report["after"]
    rows = [
        ("Total Records", b["rows"], a["rows"]),
        ("Total Columns", b["columns"], a["columns"]),
        ("Missing Values", b["missing"], a["missing"]),
        ("Duplicate Records", b["duplicates"], a["duplicates"]),
        ("Invalid Values", b["invalid"], a["invalid"]),
        ("Outliers", b["outliers"], a["outliers"]),
        ("Data Quality Score (%)", b["score"], a["score"]),
    ]
    return pd.DataFrame(rows, columns=["Data Quality Check", "Before", "After"])


def column_profile(raw_df, cleaned_df):
    """One row per column: type, missing before/after, unique values."""
    prepared, _, missing = prepare(raw_df)
    out = []
    for c in cleaned_df.columns:
        out.append({
            "Column": c,
            "Type Before": str(prepared[c].dtype) if c in prepared else "-",
            "Type After": str(cleaned_df[c].dtype),
            "Missing Before": int(missing.get(c, 0)),
            "Missing After": int(cleaned_df[c].isna().sum()),
            "Unique Values": int(cleaned_df[c].nunique()),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Data quality charts
# --------------------------------------------------------------------------
def fig_missing_by_column(report):
    s = pd.Series(report["missing_by_column"]).sort_values()
    s = s[s > 0]
    if s.empty:
        return None
    fig, ax = _fig()
    ax.barh(s.index, s.values, color=PRIMARY)
    ax.set_title("Missing Values by Column (raw data)")
    ax.set_xlabel("Missing values")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def fig_before_after(report):
    keys = ["missing", "duplicates", "invalid", "outliers"]
    labels = ["Missing", "Duplicates", "Invalid", "Outliers"]
    x = np.arange(len(keys))
    fig, ax = _fig()
    ax.bar(x - 0.2, [report["before"][k] for k in keys], 0.4, label="Before", color=ACCENT)
    ax.bar(x + 0.2, [report["after"][k] for k in keys], 0.4, label="After", color=PRIMARY)
    ax.set_xticks(x, labels)
    ax.set_title("Data Issues: Before vs After Cleaning")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_duplicates(report):
    b = report["before"]
    unique = b["rows"] - b["duplicates"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie([unique, b["duplicates"]], labels=["Unique", "Duplicate"], autopct="%1.1f%%",
           colors=[PRIMARY, ACCENT], startangle=90, wedgeprops={"width": 0.45})
    ax.set_title("Duplicate Records in Raw Data")
    return fig


def fig_dtype_distribution(report):
    before = pd.Series(report["dtypes_before"], name="Before")
    after = pd.Series(report["dtypes_after"], name="After")
    df = pd.concat([before, after], axis=1).fillna(0)
    fig, ax = _fig()
    df.plot(kind="bar", ax=ax, color=[ACCENT, PRIMARY], rot=0)
    ax.set_title("Data Type Distribution (number of columns)")
    ax.set_ylabel("Columns")
    fig.tight_layout()
    return fig


def fig_outliers(raw_df):
    prepared, _, _ = prepare(raw_df)
    num = prepared.select_dtypes(include="number")
    num = num[[c for c in num.columns if not c.endswith("_id") and "discount" not in c]]
    if num.empty:
        return None
    # Scale each column 0-1 so they fit on one chart
    scaled = (num - num.min()) / (num.max() - num.min()).replace(0, 1)
    fig, ax = _fig(8, 4)
    sns.boxplot(data=scaled, ax=ax, color="#99F6E4", flierprops={"markerfacecolor": "#E11D48"})
    ax.set_title("Outlier Detection (raw numeric columns, scaled 0-1)")
    fig.tight_layout()
    return fig


def fig_completeness(report, cleaned_df):
    rows_b = report["before"]["rows"]
    miss = pd.Series(report["missing_by_column"])
    before = 100 * (1 - miss / rows_b)
    after = 100 * (1 - cleaned_df.isna().sum() / max(len(cleaned_df), 1))
    df = pd.DataFrame({"Before": before, "After": after}).dropna()
    fig, ax = _fig(8, 4)
    df.plot(kind="bar", ax=ax, color=[ACCENT, PRIMARY], rot=45)
    ax.set_ylim(min(80, df.values.min() - 2), 100.5)
    ax.set_title("Data Completeness by Column (%)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Business charts
# --------------------------------------------------------------------------
def _cols(df):
    return {
        "revenue": find_column(df, "revenue", "sales", "amount"),
        "category": find_column(df, "category"),
        "region": find_column(df, "region"),
        "product": find_column(df, "product"),
        "date": find_column(df, "purchase_date", "date"),
        "quantity": find_column(df, "quantity", "qty"),
        "gender": find_column(df, "gender"),
        "age": find_column(df, "age"),
        "customer": find_column(df, "customer_id", "customer"),
    }


def missing_business_columns(df):
    c = _cols(df)
    return [k for k in ["revenue", "category", "region", "product", "date", "quantity"] if c[k] is None]


def fig_revenue_by(df, key):
    c = _cols(df)
    if not c[key] or not c["revenue"] or df.empty:
        return None
    s = df.groupby(c[key])[c["revenue"]].sum().sort_values()
    fig, ax = _fig()
    ax.barh(s.index.astype(str), s.values, color=PALETTE[: len(s)][::-1] if len(s) <= 7 else PRIMARY)
    ax.set_title(f"Revenue by {key.title()}")
    ax.set_xlabel("Revenue")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig


def fig_top_products(df, n=10):
    c = _cols(df)
    if not c["product"] or not c["revenue"] or df.empty:
        return None
    s = df.groupby(c["product"])[c["revenue"]].sum().nlargest(n).sort_values()
    fig, ax = _fig()
    ax.barh(s.index.astype(str), s.values, color=PRIMARY)
    ax.set_title(f"Top {n} Products by Revenue")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig


def monthly_revenue(df):
    c = _cols(df)
    if not c["date"] or not c["revenue"] or df.empty:
        return None
    d = pd.to_datetime(df[c["date"]], errors="coerce")
    return df.groupby(d.dt.to_period("M"))[c["revenue"]].sum()


def fig_monthly_trend(df):
    s = monthly_revenue(df)
    if s is None or s.empty:
        return None
    fig, ax = _fig(8, 4)
    ax.plot(s.index.astype(str), s.values, marker="o", color=PRIMARY, linewidth=2)
    ax.fill_between(s.index.astype(str), s.values, alpha=0.15, color=PRIMARY)
    ax.set_title("Monthly Revenue Trend")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig


def fig_quantity_sold(df):
    c = _cols(df)
    key = c["category"] or c["product"]
    if not key or not c["quantity"] or df.empty:
        return None
    s = df.groupby(key)[c["quantity"]].sum().sort_values(ascending=False)
    fig, ax = _fig()
    sns.barplot(x=s.index.astype(str), y=s.values, ax=ax, color=PALETTE[1])
    ax.set_title(f"Quantity Sold by {key.replace('_', ' ').title()}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_customer_distribution(df):
    c = _cols(df)
    if df.empty or not (c["gender"] or c["age"]):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if c["gender"]:
        g = df[c["gender"]].value_counts()
        axes[0].pie(g.values, labels=g.index, autopct="%1.0f%%", colors=PALETTE, startangle=90)
        axes[0].set_title("Customers by Gender")
    else:
        axes[0].axis("off")
    if c["age"]:
        sns.histplot(df[c["age"]], bins=15, ax=axes[1], color=PRIMARY)
        axes[1].set_title("Customer Age Distribution")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Automated insights (always calculated from the data)
# --------------------------------------------------------------------------
def generate_insights(df, report):
    b, a = report["before"], report["after"]
    ins = [
        f"The raw dataset had {b['rows']:,} records and {b['columns']} columns; "
        f"{a['rows']:,} clean records remain ({report['removed_rows']} removed, {report['cleaning_pct']}%).",
        f"{report['missing_pct']}% of all cells were missing ({b['missing']:,} values); all were handled.",
        f"{b['duplicates']} duplicate records were removed.",
        f"{b['invalid']} invalid values and {b['outliers']} outliers were detected and fixed.",
        f"Data quality score improved from {b['score']}% to {a['score']}%.",
    ]
    c = _cols(df)
    if df.empty:
        return ins + ["No rows match the current filters."]
    rev = c["revenue"]
    if rev:
        total = df[rev].sum()
        ins.append(f"Total revenue is {total:,.2f} with an average order value of {df[rev].mean():,.2f}.")
        for key in ["category", "region"]:
            if c[key]:
                s = df.groupby(c[key])[rev].sum().sort_values(ascending=False)
                ins.append(f"Highest-revenue {key}: {s.index[0]} ({s.iloc[0]:,.2f}, "
                           f"{100 * s.iloc[0] / total:.1f}% of total). Lowest: {s.index[-1]}.")
        if c["product"]:
            s = df.groupby(c["product"])[rev].sum()
            ins.append(f"Top product by revenue: {s.idxmax()} ({s.max():,.2f}).")
    if c["product"] and c["quantity"]:
        q = df.groupby(c["product"])[c["quantity"]].sum()
        ins.append(f"Most units sold: {q.idxmax()} ({int(q.max()):,} units).")
    m = monthly_revenue(df)
    if m is not None and len(m) >= 2:
        slope = np.polyfit(np.arange(len(m)), m.values, 1)[0]
        trend = "upward" if slope > 0 else "downward"
        ins.append(f"Monthly revenue shows an overall {trend} trend. Best month: {m.idxmax()} "
                   f"({m.max():,.2f}); weakest month: {m.idxmin()} ({m.min():,.2f}).")
    if c["gender"] and rev:
        g = df.groupby(c["gender"])[rev].sum()
        ins.append(f"{g.idxmax()} customers generated the most revenue ({100 * g.max() / g.sum():.1f}%).")
    if c["customer"]:
        ins.append(f"There are {df[c['customer']].nunique():,} unique customers.")
    return ins


# --------------------------------------------------------------------------
# Downloadable reports
# --------------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(report, insights, figures, profile_df):
    """Self-contained HTML report (charts embedded) that opens in any browser."""
    table = quality_summary_table(report).to_html(index=False, classes="t")
    prof = profile_df.to_html(index=False, classes="t")
    ops = "".join(f"<li>{o}</li>" for o in report["operations"])
    ins = "".join(f"<li>{i}</li>" for i in insights)
    imgs = "".join(
        f"<figure><img src='data:image/png;base64,{fig_to_base64(f)}'/><figcaption>{t}</figcaption></figure>"
        for t, f in figures.items() if f is not None
    )
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Automated Data Cleaning Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1000px;margin:auto;padding:30px;color:#1e293b}}
h1{{color:#0F766E;border-bottom:3px solid #0F766E;padding-bottom:8px}} h2{{color:#0F766E;margin-top:32px}}
.t{{border-collapse:collapse;width:100%}} .t th{{background:#0F766E;color:#fff;padding:8px}}
.t td{{border:1px solid #e2e8f0;padding:6px 8px}} .t tr:nth-child(even){{background:#f1f5f9}}
figure{{display:inline-block;width:48%;margin:1%;vertical-align:top}} img{{width:100%}}
figcaption{{text-align:center;color:#64748b;font-size:13px}} li{{margin:4px 0}}
</style></head><body>
<h1>Automated Data Cleaning &amp; Reporting</h1>
<p>Generated on {datetime.now():%d %B %Y, %H:%M}</p>
<h2>1. Dataset Summary</h2>{table}
<h2>2. Cleaning Operations Performed</h2><ol>{ops}</ol>
<h2>3. Column Profile</h2>{prof}
<h2>4. Key Business Insights</h2><ul>{ins}</ul>
<h2>5. Charts and Visualizations</h2>{imgs}
</body></html>"""


def build_excel_report(report, insights, profile_df, cleaned_df=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        quality_summary_table(report).to_excel(xw, sheet_name="Quality Summary", index=False)
        profile_df.to_excel(xw, sheet_name="Column Profile", index=False)
        pd.DataFrame({"Cleaning Operation": report["operations"]}).to_excel(xw, sheet_name="Cleaning Log", index=False)
        pd.DataFrame({"Insight": insights}).to_excel(xw, sheet_name="Insights", index=False)
        if cleaned_df is not None:
            cleaned_df.to_excel(xw, sheet_name="Cleaned Data", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 90)
    return buf.getvalue()


def df_to_excel_bytes(df, sheet="Data"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=sheet, index=False)
    return buf.getvalue()
