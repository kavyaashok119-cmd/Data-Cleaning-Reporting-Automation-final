"""
cleaning.py
-----------
Reusable, automated data-cleaning pipeline.

Main entry points
    clean_data(df)              -> cleaned DataFrame
    clean_data_with_report(df)  -> (cleaned DataFrame, report dict)
    generate_sample_data()      -> realistic "dirty" sales dataset for demos

The pipeline does not depend on fixed column names. It works out what each
column holds (IDs, dates, numbers, text) from the column name and its values,
so it also works on similar datasets.
"""

import os
import re

import numpy as np
import pandas as pd

# Text values that should be treated as "missing"
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "missing", "nat"}

# Different spellings of gender mapped to one standard value
GENDER_MAP = {
    "m": "Male", "male": "Male", "man": "Male",
    "f": "Female", "female": "Female", "woman": "Female",
    "o": "Other", "other": "Other",
}

# Common spelling variants of categories / regions
VALUE_FIXES = {
    "Electronic": "Electronics", "Elec": "Electronics",
    "Clothes": "Clothing", "Cloth": "Clothing",
    "Home And Kitchen": "Home & Kitchen", "Home-Kitchen": "Home & Kitchen",
    "N": "North", "S": "South", "E": "East", "W": "West",
}

# Columns containing these words can never be negative
NON_NEGATIVE_KEYWORDS = ["quantity", "qty", "price", "revenue", "amount", "sales", "cost", "discount"]

# Multiplier for the IQR outlier rule (3.0 = only extreme values)
OUTLIER_K = 3.0


# --------------------------------------------------------------------------
# Small helpers that work out what a column holds
# --------------------------------------------------------------------------
def _is_id_col(col):
    return col == "id" or col.endswith("_id")


def _is_date_col(col):
    return "date" in col


def _is_age_col(col):
    return col == "age" or col.endswith("_age")


def _is_non_negative(col):
    return _is_age_col(col) or any(k in col for k in NON_NEGATIVE_KEYWORDS)


def find_column(df, *keywords):
    """Return the first column whose name contains one of the keywords."""
    for k in keywords:
        for c in df.columns:
            if c == k:
                return c
    for k in keywords:
        for c in df.columns:
            if k in c:
                return c
    return None


# --------------------------------------------------------------------------
# Step 1: standardize column names
# --------------------------------------------------------------------------
def standardize_column_names(df):
    """'CustomerID' -> 'customer_id', ' Purchase Date ' -> 'purchase_date'."""
    df = df.copy()
    new_cols, seen = [], {}
    for c in df.columns:
        name = str(c).strip()
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)  # CamelCase -> snake
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or "column"
        if name in seen:  # avoid duplicate column names
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


# --------------------------------------------------------------------------
# Step 2: trim spaces, mark missing tokens and convert data types
# --------------------------------------------------------------------------
def prepare(raw_df):
    """
    Standardize names, trim spaces, turn tokens like 'N/A' into NaN, and
    convert columns to the right types.

    Returns (df, conversion_failures, missing_by_column)
    conversion_failures = values that had content but could not be read as a
    valid date or number (counted as invalid values).
    """
    df = standardize_column_names(raw_df)

    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        s = s.mask(s.str.lower().isin(MISSING_TOKENS))
        df[c] = s.astype(object).where(s.notna(), np.nan)

    missing_by_col = df.isna().sum().astype(int)
    failures = {}

    for c in df.columns:
        if _is_id_col(c):
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
            df.loc[df[c].notna(), c] = df.loc[df[c].notna(), c].astype(str)
            continue
        if _is_date_col(c):
            before = df[c].notna().sum()
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                df[c] = pd.to_datetime(df[c].astype(str), errors="coerce")
            failed = int(before - df[c].notna().sum())
            if failed:
                failures[c] = failed
        elif df[c].dtype == object:
            text = df[c].astype(str).str.replace(r"[$,€£₹%\s]", "", regex=True)
            nums = pd.to_numeric(text.where(df[c].notna()), errors="coerce")
            non_null = df[c].notna().sum()
            # Treat it as a number column if at least 80% of values are numbers
            if non_null and nums.notna().sum() / non_null >= 0.8:
                failed = int(non_null - nums.notna().sum())
                if failed:
                    failures[c] = failed
                df[c] = nums

    return df, failures, missing_by_col


# --------------------------------------------------------------------------
# Step 3: standardize text values
# --------------------------------------------------------------------------
def standardize_text(df):
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        s = df[c]
        mask = s.notna()
        if _is_id_col(c):
            df.loc[mask, c] = s[mask].astype(str).str.upper()
        elif "gender" in c or c == "sex":
            low = s[mask].astype(str).str.lower()
            df.loc[mask, c] = low.map(GENDER_MAP).fillna(low.str.title())
        else:
            df.loc[mask, c] = s[mask].astype(str).str.title().replace(VALUE_FIXES)
    return df


# --------------------------------------------------------------------------
# Step 4: detect and fix invalid values
# --------------------------------------------------------------------------
def detect_invalid(df):
    """Count invalid numbers per column (negatives, impossible ages, bad discounts)."""
    found = {}
    for c in df.select_dtypes(include="number").columns:
        if _is_id_col(c):
            continue
        s = df[c]
        bad = pd.Series(False, index=s.index)
        if "discount" in c:
            bad |= (s < 0) | (s > 1)
        elif _is_non_negative(c):
            bad |= s < 0
        if _is_age_col(c):
            bad |= s > 110
        if bad.sum():
            found[c] = int(bad.sum())
    return found


def fix_invalid(df):
    """Fix invalid values where possible, otherwise set them to NaN to be filled later."""
    df = df.copy()
    for c in df.select_dtypes(include="number").columns:
        if _is_id_col(c):
            continue
        if "discount" in c:
            pct = (df[c] > 1) & (df[c] <= 100)  # 15 means 15% -> 0.15
            df.loc[pct, c] = df.loc[pct, c] / 100
            bad = (df[c] < 0) | (df[c] > 1)
        else:
            bad = (df[c] < 0) if _is_non_negative(c) else pd.Series(False, index=df.index)
            if _is_age_col(c):
                bad |= df[c] > 110
        df.loc[bad, c] = np.nan
    return df


# --------------------------------------------------------------------------
# Step 5: detect and cap outliers (IQR rule)
# --------------------------------------------------------------------------
def _iqr_bounds(s, k, non_negative):
    """IQR limits for one series (returns None if it can't be computed)."""
    s = s.dropna()
    if s.nunique() < 5:
        return None
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (max(lo, 0) if non_negative else lo), hi


def detect_outliers(df, k=OUTLIER_K):
    """
    Return {column: (lower_limits, upper_limits, count)} using the IQR rule.
    Limits are Series, so prices can be checked per product (a laptop is not
    an outlier just because it costs more than a notebook).
    Derived totals (revenue/sales/amount) are skipped - they are recalculated.
    """
    result = {}
    product_col = find_column(df, "product")
    for c in df.select_dtypes(include="number").columns:
        if _is_id_col(c) or "discount" in c or any(w in c for w in ("revenue", "sales", "amount")):
            continue
        nn = _is_non_negative(c)
        lo = pd.Series(np.nan, index=df.index)
        hi = pd.Series(np.nan, index=df.index)
        if "price" in c and product_col and product_col != c:
            for _, idx in df.groupby(product_col).groups.items():
                b = _iqr_bounds(df.loc[idx, c], k, nn)
                if b:
                    lo[idx], hi[idx] = b
        else:
            b = _iqr_bounds(df[c], k, nn)
            if b:
                lo[:], hi[:] = b
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        if lo.notna().any():
            result[c] = (lo, hi, n)
    return result


# --------------------------------------------------------------------------
# Step 6: fill missing values
# --------------------------------------------------------------------------
def fill_missing(df):
    df = df.copy()
    log = {}
    product_col = find_column(df, "product")
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            whole = bool((df[c].dropna() % 1 == 0).all())
            if "price" in c and product_col and product_col != c:
                df[c] = df[c].fillna(df.groupby(product_col)[c].transform("median"))
                method = f"median per {product_col}"
            else:
                method = "median"
            df[c] = df[c].fillna(df[c].median())
            if whole:
                df[c] = df[c].round().astype("int64")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            continue  # rows without a date are dropped earlier
        elif _is_id_col(c):
            df[c] = df[c].fillna("UNKNOWN")
            method = "'UNKNOWN'"
        else:
            mode = df[c].mode()
            if df[c].nunique() <= 20 and not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
                method = f"most common value ('{mode.iloc[0]}')"
            else:
                df[c] = df[c].fillna("Unknown")
                method = "'Unknown'"
        log[c] = (n, method)
    return df, log


# --------------------------------------------------------------------------
# Profiling and quality score
# --------------------------------------------------------------------------
def quality_score(rows, cols, missing, duplicates, invalid):
    """Score from 0 to 100: 50% completeness, 25% uniqueness, 25% validity."""
    if rows == 0 or cols == 0:
        return 0.0
    completeness = 1 - missing / (rows * cols)
    uniqueness = 1 - duplicates / rows
    validity = max(0.0, 1 - invalid / rows)
    return round(100 * (0.5 * completeness + 0.25 * uniqueness + 0.25 * validity), 1)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def clean_data_with_report(raw_df):
    """Run the full cleaning pipeline. Returns (cleaned_df, report)."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The dataset is empty, so there is nothing to clean.")

    ops = []
    raw_rows, raw_cols = raw_df.shape

    # 1-2. Names, spaces, missing tokens, data types
    df, failures, missing_by_col = prepare(raw_df)
    ops.append("Standardized column names (e.g. 'CustomerID' -> 'customer_id').")
    ops.append("Removed extra spaces and turned text like 'N/A', 'null', '?' into missing values.")
    dtypes_before = raw_df.dtypes.astype(str).value_counts().to_dict()
    ops.append("Converted columns to the right data types (numbers, dates, text).")

    missing_before = int(missing_by_col.sum())
    invalid_by_col = detect_invalid(df)
    for c, n in failures.items():
        invalid_by_col[c] = invalid_by_col.get(c, 0) + n
    invalid_before = int(sum(invalid_by_col.values()))
    if failures:
        ops.append(f"Found {sum(failures.values())} unreadable dates/numbers: {failures}.")

    # 3. Text standardization
    df = standardize_text(df)
    ops.append("Standardized text (Title Case, unified gender values, fixed category/region spellings).")

    # 4. Duplicates (checked after text clean-up, so ' north' and 'North' match)
    duplicates_before = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    ops.append(f"Removed {duplicates_before} duplicate records.")

    # 5. Invalid values
    df = fix_invalid(df)
    if invalid_before:
        ops.append(f"Fixed {invalid_before} invalid values (negatives, impossible ages, wrong discount format).")

    # 6. Outliers
    outliers = detect_outliers(df)
    outliers_before = int(sum(v[2] for v in outliers.values()))
    for c, (lo, hi, n) in outliers.items():
        if n:
            df[c] = df[c].clip(lower=lo.fillna(-np.inf), upper=hi.fillna(np.inf))
    if outliers_before:
        ops.append(f"Capped {outliers_before} extreme outliers using the IQR rule (k={OUTLIER_K}).")

    # 7. Missing values
    rows_before = len(df)
    df = df.dropna(how="all")
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        df = df.dropna(subset=date_cols)
    dropped_rows = rows_before - len(df)
    if dropped_rows:
        ops.append(f"Removed {dropped_rows} rows with a missing or invalid date.")
    df, fill_log = fill_missing(df)
    for c, (n, m) in fill_log.items():
        ops.append(f"Filled {n} missing values in '{c}' with {m}.")

    # 8. Recalculate revenue so it matches quantity x price x (1 - discount)
    q, p = find_column(df, "quantity", "qty"), find_column(df, "price")
    r, d = find_column(df, "revenue", "sales", "amount"), find_column(df, "discount")
    if q and p and r and r not in (q, p):
        disc = df[d] if d else 0
        df[r] = (df[q] * df[p] * (1 - disc)).round(2)
        ops.append(f"Recalculated '{r}' as {q} x {p}" + (f" x (1 - {d})." if d else "."))

    # 9. Final check for duplicates created by the fixes
    extra = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_before += extra

    rows_after, cols_after = df.shape
    missing_after = int(df.isna().sum().sum())
    invalid_after = int(sum(detect_invalid(df).values()))
    removed = raw_rows - rows_after

    report = {
        "before": {
            "rows": raw_rows, "columns": raw_cols, "missing": missing_before,
            "duplicates": duplicates_before, "invalid": invalid_before,
            "outliers": outliers_before,
            "score": quality_score(raw_rows, raw_cols, missing_before, duplicates_before, invalid_before),
        },
        "after": {
            "rows": rows_after, "columns": cols_after, "missing": missing_after,
            "duplicates": 0, "invalid": invalid_after,
            "outliers": int(sum(v[2] for v in detect_outliers(df).values())),
            "score": quality_score(rows_after, cols_after, missing_after, 0, invalid_after),
        },
        "removed_rows": removed,
        "cleaning_pct": round(100 * removed / raw_rows, 2) if raw_rows else 0.0,
        "missing_pct": round(100 * missing_before / (raw_rows * raw_cols), 2),
        "missing_by_column": missing_by_col.to_dict(),
        "invalid_by_column": invalid_by_col,
        "outliers_by_column": {c: v[2] for c, v in outliers.items() if v[2]},
        "dtypes_before": dtypes_before,
        "dtypes_after": df.dtypes.astype(str).value_counts().to_dict(),
        "operations": ops,
    }
    return df, report


def clean_data(df):
    """Reusable cleaning function: give it a raw DataFrame, get a cleaned one back."""
    cleaned, _ = clean_data_with_report(df)
    return cleaned


# --------------------------------------------------------------------------
# Sample dataset with deliberate data-quality problems
# --------------------------------------------------------------------------
def generate_sample_data(n=520, n_duplicates=30, seed=42):
    rng = np.random.default_rng(seed)
    products = {
        "Laptop": ("Electronics", 850), "Smartphone": ("Electronics", 600),
        "Headphones": ("Electronics", 80), "Tablet": ("Electronics", 400),
        "Smartwatch": ("Electronics", 220), "T-Shirt": ("Clothing", 20),
        "Jeans": ("Clothing", 45), "Jacket": ("Clothing", 90),
        "Sneakers": ("Clothing", 70), "Blender": ("Home & Kitchen", 60),
        "Coffee Maker": ("Home & Kitchen", 95), "Cookware Set": ("Home & Kitchen", 120),
        "Desk Lamp": ("Home & Kitchen", 35), "Novel": ("Books", 15),
        "Textbook": ("Books", 65), "Yoga Mat": ("Sports", 30),
        "Dumbbells": ("Sports", 55), "Football": ("Sports", 25),
    }
    first = ["Aarav", "Priya", "Rahul", "Ananya", "John", "Emma", "Liam", "Sofia", "Arjun",
             "Meera", "David", "Olivia", "Vikram", "Sneha", "Michael", "Aisha", "Rohan", "Grace"]
    last = ["Sharma", "Patel", "Singh", "Smith", "Brown", "Khan", "Iyer", "Garcia",
            "Wilson", "Reddy", "Mehta", "Taylor", "Nair", "Lee"]
    regions = ["North", "South", "East", "West", "Central"]

    customers = {
        f"C{1000 + i}": (f"{rng.choice(first)} {rng.choice(last)}",
                         str(rng.choice(["Male", "Female"])), int(rng.integers(18, 70)))
        for i in range(300)
    }
    cust_ids = list(customers)
    names = list(products)
    start = pd.Timestamp("2024-01-01")

    rows = []
    for _ in range(n):
        cid = cust_ids[rng.integers(len(cust_ids))]
        cname, gender, age = customers[cid]
        prod = names[rng.integers(len(names))]
        cat, base = products[prod]
        price = round(base * rng.uniform(0.85, 1.15), 2)
        qty = int(rng.integers(1, 8))
        disc = float(rng.choice([0, 0.05, 0.10, 0.15, 0.20]))
        date = start + pd.Timedelta(days=int(rng.integers(0, 366)))
        rows.append({
            "CustomerID": cid, "CustomerName": cname, "Gender": gender, "Age": age,
            "Product": prod, "Category": cat, "Region": str(rng.choice(regions)),
            "Quantity": qty, "Price": price, "Discount": disc,
            "Revenue": round(qty * price * (1 - disc), 2),
            "PurchaseDate": date.strftime("%Y-%m-%d"),
        })
    df = pd.DataFrame(rows).astype(object)

    def pick(k):
        return rng.choice(n, size=k, replace=False)

    # Missing values
    for col, k in {"Age": 25, "Gender": 15, "Region": 15, "Price": 12,
                   "CustomerName": 10, "Discount": 10, "PurchaseDate": 5}.items():
        df.loc[pick(k), col] = np.nan
    df.loc[pick(8), "Region"] = "N/A"

    # Inconsistent text values
    for i in pick(45):
        if pd.notna(df.at[i, "Gender"]):
            df.at[i, "Gender"] = str(rng.choice(["M", "F", "male", "FEMALE", " Female ", "m"]))
    for i in pick(45):
        if pd.notna(df.at[i, "Region"]) and df.at[i, "Region"] != "N/A":
            r = df.at[i, "Region"]
            df.at[i, "Region"] = str(rng.choice([r.lower(), r.upper(), f"  {r} "]))
    for i in pick(35):
        c = df.at[i, "Category"]
        variants = [c.lower(), c.upper(), f" {c}"]
        if c == "Electronics":
            variants.append("Electronic")
        if c == "Clothing":
            variants.append("Clothes")
        df.at[i, "Category"] = str(rng.choice(variants))
    for i in pick(25):
        df.at[i, "Product"] = str(rng.choice([df.at[i, "Product"].lower(), f" {df.at[i, 'Product']}  "]))
    for i in pick(20):
        if pd.notna(df.at[i, "CustomerName"]):
            df.at[i, "CustomerName"] = f"  {df.at[i, 'CustomerName'].lower()} "

    # Incorrect formats
    for i in pick(25):
        if pd.notna(df.at[i, "Price"]):
            df.at[i, "Price"] = f"${df.at[i, 'Price']:,.2f}"
    for i in pick(40):
        if pd.notna(df.at[i, "PurchaseDate"]):
            d = pd.Timestamp(df.at[i, "PurchaseDate"])
            df.at[i, "PurchaseDate"] = d.strftime(str(rng.choice(["%B %d, %Y", "%Y/%m/%d", "%d-%b-%Y"])))
    df.loc[pick(6), "PurchaseDate"] = ["2024-13-45", "not a date", "31/31/2024", "2024-02-30", "unknown", "??"]

    # Invalid values
    df.loc[pick(4), "Age"] = 150
    df.loc[pick(4), "Age"] = -5
    for i in pick(8):
        df.at[i, "Quantity"] = -int(rng.integers(1, 5))
    for i in pick(4):
        if isinstance(df.at[i, "Price"], float):
            df.at[i, "Price"] = -df.at[i, "Price"]
    df.loc[pick(6), "Discount"] = 15  # percent instead of fraction

    # Outliers
    df.loc[pick(5), "Quantity"] = 250

    # Duplicate records
    dups = df.iloc[rng.choice(n, size=n_duplicates, replace=False)]
    df = pd.concat([df, dups], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    # Run "python cleaning.py" to create raw_data.csv and cleaned_data.csv
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base, "data"), exist_ok=True)
    raw = generate_sample_data()
    raw.to_csv(os.path.join(base, "data", "raw_data.csv"), index=False)
    cleaned, rep = clean_data_with_report(pd.read_csv(os.path.join(base, "data", "raw_data.csv")))
    cleaned.to_csv(os.path.join(base, "data", "cleaned_data.csv"), index=False)
    print(f"Raw rows: {rep['before']['rows']}  ->  Clean rows: {rep['after']['rows']}")
    for op in rep["operations"]:
        print(" -", op)
