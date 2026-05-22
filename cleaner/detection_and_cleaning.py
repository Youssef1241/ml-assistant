import pandas as pd
import numpy as np
from collections import defaultdict

def make_report(state: dict) -> str:
    df = pd.read_csv(state["df_info"]["filepath"])
    issues = detect_type_issues(df)
    sentinels = detect_sentinels(df)
    final_report = {"issues": issues, "sentinels": sentinels}
    df_info = state['df_info']
    df_info.update({"issues": final_report})
    return {
        "df_info": df_info,
        }
    
# def find_issues(df: pd.DataFrame) -> dict:
#     """Find issues in the data and return a dictionary of the issues"""
#     issues = {}
#     def report_duplicates(df):
#         total = len(df)
#         n_exact = df.duplicated().sum()
#         pct = (n_exact / total) * 100
#         return "Total duplicates: " + str(n_exact) + " (" + str(pct) + "%)"
#     issues["duplicates"] = report_duplicates(df)
#     return issues

KNOWN_NUMERIC_SENTINELS = {-1, 0, 999, 9999, -9999, 99, -99, 999999, -999999, 9999999}
KNOWN_STRING_SENTINELS = {
    "n/a", "na", "none", "null", "nil", "missing", "unknown",
    "?", "-", "--", "nan", "undefined", "not available", "n.a.", ""
}

def detect_sentinels(df, freq_threshold=0.05, z_threshold=4.0):
    """
    Detects potential sentinel values in all columns.
    
    freq_threshold : flag a value if it appears in more than this % of rows
                     AND is a known sentinel pattern
    z_threshold    : z-score beyond which a numeric value is considered an outlier
    
    Returns a dict: { col_name: [ {value, reason, count, pct} ] }
    """
    report = defaultdict(list)

    for col in df.columns:
        series = df[col]
        total = len(series)
        value_counts = series.value_counts(dropna=False)

        # ── Numeric columns ──────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            col_mean = series.mean()
            col_std  = series.std()

            for val, count in value_counts.items():
                if pd.isna(val):
                    continue

                pct = count / total
                reasons = []

                # Check 1: known sentinel number
                if val in KNOWN_NUMERIC_SENTINELS:
                    reasons.append(f"known sentinel number ({val})")

                # Check 2: extreme z-score (statistical outlier)
                if col_std > 0:
                    z = abs((val - col_mean) / col_std)
                    if z > z_threshold:
                        reasons.append(f"extreme outlier (z={z:.1f})")

                # Check 3: high frequency outlier
                # i.e. one specific extreme value repeats suspiciously often
                if pct >= freq_threshold and val in KNOWN_NUMERIC_SENTINELS:
                    reasons.append(f"high frequency ({pct:.1%} of rows)")

                if reasons:
                    report[col].append({
                        "value"  : val,
                        "count"  : count,
                        "pct"    : f"{pct:.1%}",
                        "reasons": ", ".join(reasons)
                    })

        # ── String / categorical columns ─────────────────────────────
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            for val, count in value_counts.items():
                pct = count / total
                reasons = []

                if pd.isna(val):
                    continue

                normalized = str(val).strip().lower()

                # Check 1: known string sentinel
                if normalized in KNOWN_STRING_SENTINELS:
                    reasons.append(f"known string sentinel ('{val}')")

                # Check 2: very short suspicious strings
                if normalized in {"?", "-", ".", "_", "x", "xx", "xxx"}:
                    reasons.append(f"suspicious placeholder ('{val}')")

                if reasons:
                    report[col].append({
                        "value"  : val,
                        "count"  : count,
                        "pct"    : f"{pct:.1%}",
                        "reasons": ", ".join(reasons)
                    })

    return dict(report)

import re

def detect_type_issues(df, sample_threshold=0.85):
    """
    Scans all columns and flags ones where the actual content
    doesn't match the stored dtype.
    
    sample_threshold: % of non-null values that must match a pattern
                      to confidently flag it (default 85%)
    """
    issues = {}

    for col in df.columns:
        series      = df[col].dropna()
        current     = df[col].dtype
        col_issues  = []

        if len(series) == 0:
            continue

        # ── 1. Object columns that look numeric ──────────────────────
        if current == object:
            # Strip common noise before testing
            cleaned = (
                series.astype(str)
                      .str.strip()
                      .str.replace(r"^[\$£€]", "", regex=True)   # currency
                      .str.replace(",", "")                        # thousands separator
                      .str.replace(r"%$", "", regex=True)         # percentage
            )

            numeric_ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()

            if numeric_ratio >= sample_threshold:
                # Further distinguish int vs float
                is_float = cleaned.str.contains(r"\.").any()
                suggested = "float64" if is_float else "int64"

                # Check if it was actually a percentage
                has_pct = series.astype(str).str.strip().str.endswith("%").mean()
                note = " (looks like percentage — consider dividing by 100)" if has_pct > 0.5 else ""

                col_issues.append({
                    "issue"    : "numeric stored as string",
                    "suggested": suggested,
                    "note"     : note,
                    "confidence": f"{numeric_ratio:.0%}"
                })

        # ── 2. Object columns that look like dates ───────────────────
        if current == object:
            date_ratio = pd.to_datetime(
                series.astype(str).str.strip(),
                errors="coerce",
                infer_datetime_format=True
            ).notna().mean()

            if date_ratio >= sample_threshold:
                col_issues.append({
                    "issue"    : "date/datetime stored as string",
                    "suggested": "datetime64",
                    "note"     : "check for ambiguous formats like 01/02/03",
                    "confidence": f"{date_ratio:.0%}"
                })

        # ── 3. Object columns that look like booleans ────────────────
        if current == object:
            bool_map = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
            bool_ratio = series.astype(str).str.strip().str.lower().isin(bool_map).mean()

            if bool_ratio >= sample_threshold:
                col_issues.append({
                    "issue"    : "boolean stored as string",
                    "suggested": "bool",
                    "note"     : "map to True/False explicitly",
                    "confidence": f"{bool_ratio:.0%}"
                })

        # ── 4. Numeric columns that look like IDs ────────────────────
        if pd.api.types.is_integer_dtype(current):
            unique_ratio = series.nunique() / len(series)

            if unique_ratio > 0.95:
                col_issues.append({
                    "issue"    : "ID/key stored as integer",
                    "suggested": "string/object",
                    "note"     : "IDs shouldn't be numeric — math on them is meaningless",
                    "confidence": f"{unique_ratio:.0%} unique"
                })

        # ── 5. Float columns that should be int ──────────────────────
        if current == "float64":
            is_whole = (series % 1 == 0).mean()

            if is_whole >= sample_threshold:
                col_issues.append({
                    "issue"    : "integer stored as float",
                    "suggested": "int64 (or Int64 if nulls exist)",
                    "note"     : "likely caused by NaNs forcing float dtype",
                    "confidence": f"{is_whole:.0%}"
                })

        # ── 6. Object columns with very low cardinality ──────────────
        if current == object:
            unique_ratio = series.nunique() / len(series)

            if unique_ratio < 0.05:
                col_issues.append({
                    "issue"    : "low-cardinality string — candidate for category dtype",
                    "suggested": "category",
                    "note"     : f"only {series.nunique()} unique values — saves memory",
                    "confidence": f"{unique_ratio:.0%} unique"
                })

        if col_issues:
            issues[col] = {
                "current_dtype": str(current),
                "issues"       : col_issues
            }

    return issues

# def apply_replacements(df, approved):
#     """Replaces approved sentinel values with NaN."""
#     df_clean = df.copy()
#     total_replaced = 0

#     for col, values in approved.items():
#         before = df_clean[col].isna().sum()
#         df_clean[col] = df_clean[col].replace(values, np.nan)
#         after = df_clean[col].isna().sum()
#         replaced = after - before
#         total_replaced += replaced
#         print(f"  🔄 '{col}': {replaced} value(s) replaced with NaN")

#     print(f"\n✅ Done. Total replacements: {total_replaced}")
#     return df_clean

