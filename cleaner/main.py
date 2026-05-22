import pandas as pd
from pandas import DataFrame

def clean_data(state: dict):
    df = pd.read_csv(state["df_info"]["filepath"])
    user_choice = state["user_choice"]['cleaning']['cleaning']
    drop_cols = user_choice.get("drop_columns", [])
    drop_outliers = user_choice.get("drop_outliers", [])
    cast_cols = user_choice.get("cast_columns", [])
    replace_sentinels = user_choice.get("replace_sentinels", [])
    if drop_cols:
        df = _drop_column(df, drop_cols)
    if drop_outliers:
        df = _drop_outliers(df, drop_outliers)
    if cast_cols:
        issues = state['df_info']['issues']['issues']
        df = _cast_to_dtypes(df, cast_cols, issues)
    if replace_sentinels:
        sentinels = state['df_info']['issues']['sentinels']
        df = _replace_sentinels(df, replace_sentinels, sentinels)
    df.to_csv(state["df_info"]["filepath"], index=False)
    return state

def _drop_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop rows with outlier values across all numeric columns using IQR method."""
    mask = pd.Series(True, index=df.index)

    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)

    return df[mask]

def _replace_sentinels(df: pd.DataFrame, columns: list, sentinel_dict: dict) -> pd.DataFrame:
    """Replace sentinel values with NaN"""
    for col in columns:
        df[col] = df[col].replace(sentinel_dict[col], pd.NA)

    return df

def _cast_to_dtypes(df: pd.DataFrame, columns: list, issues: dict) -> pd.DataFrame:
    """Cast columns to the correct dtype"""

    df_clean = df.copy()
    for col in columns:
        info = issues[col]
        for issue_info in info["issues"]:
            issue = issue_info["issue"]
            cast_fn = CAST_FUNCTIONS.get(issue)
            try:
                casted, actual_type = cast_fn(df_clean[col], col)
                df_clean[col] = casted
            except Exception as e:
                print(f"   ❌ Cast failed for '{col}': {e}")
    return df_clean

def _drop_column(df: DataFrame, columns: list,) -> (str, DataFrame):
    """Drop a column from the dataset."""
    try:
        loglist = list()
        drop_columns = list()
        for column in columns:
            if column not in df.columns:
                loglist.append( f"Column '{column}' does not exist in data.")
                continue
            drop_columns.append(column)
        df.drop(columns=drop_columns, inplace=True)
        loglist.append( f"Columns '{drop_columns}' dropped successfully.")
        return df
    except Exception as e:
        return f"Error dropping columns '{drop_columns}': {e}"


import pandas as pd
import numpy as np

# ── Individual casting functions ──────────────────────────────────────

def cast_to_numeric(series, col):
    """Strips currency, commas, % then casts to int or float."""
    cleaned = (
        series.astype(str)
              .str.strip()
              .str.replace(r"^[\$£€]", "", regex=True)
              .str.replace(",", "")
              .str.replace(r"%$", "", regex=True)
    )
    casted = pd.to_numeric(cleaned, errors="coerce")

    # Downcast to int if no decimals (and no nulls)
    if casted.notna().all() and (casted % 1 == 0).all():
        return casted.astype("int64"), "int64"
    elif casted.isna().any() and (casted % 1 == 0).all():
        return casted.astype("Int64"), "Int64 (nullable int)"  # capital I = nullable
    else:
        return casted, "float64"


def cast_to_datetime(series, col):
    """Parses strings to datetime, warns on ambiguous formats."""
    casted = pd.to_datetime(
        series.astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True
    )
    failed = casted.isna().sum() - series.isna().sum()
    if failed > 0:
        print(f"   ⚠️  {failed} value(s) in '{col}' couldn't be parsed → set to NaT")
    return casted, "datetime64[ns]"


def cast_to_bool(series, col):
    """Maps common truthy/falsy strings to boolean."""
    bool_map = {
        "true": True,  "false": False,
        "yes" : True,  "no"   : False,
        "y"   : True,  "n"    : False,
        "1"   : True,  "0"    : False,
        "t"   : True,  "f"    : False,
    }
    casted = series.astype(str).str.strip().str.lower().map(bool_map)
    failed = casted.isna().sum() - series.isna().sum()
    if failed > 0:
        print(f"   ⚠️  {failed} value(s) in '{col}' didn't match bool map → set to NaN")
    return casted, "bool"


def cast_to_string(series, col):
    """Casts to string (for IDs and similar)."""
    return series.astype(str), "object (string)"


def cast_float_to_int(series, col):
    """Converts whole-number floats to int."""
    if series.isna().any():
        return series.astype("Int64"), "Int64 (nullable int)"
    return series.astype("int64"), "int64"


def cast_to_category(series, col):
    """Converts low-cardinality string column to category dtype."""
    return series.astype("category"), "category"


# ── Casting dispatcher ────────────────────────────────────────────────

CAST_FUNCTIONS = {
    "numeric stored as string"              : cast_to_numeric,
    "date/datetime stored as string"        : cast_to_datetime,
    "boolean stored as string"              : cast_to_bool,
    "ID/key stored as integer"              : cast_to_string,
    "integer stored as float"               : cast_float_to_int,
    "low-cardinality string — candidate for category dtype": cast_to_category,
}


# ── Approval + casting pipeline ───────────────────────────────────────

def prompt_and_cast(df, issues, state):
    """
    Shows the user each detected type issue and asks for approval.
    Applies approved casts and returns the cleaned df + a change log.
    """
    df_clean = df.copy()

    target_col = state['df_info']["target_col"]
    for col, info in issues.items():
        for issue_info in info["issues"]:
            issue     = issue_info["issue"]

            if col == target_col:
                print(f"   ⏭️  Skipped '{col}'.")
                continue

            # Look up and apply the right cast function
            cast_fn = CAST_FUNCTIONS.get(issue)

            try:
                casted, actual_type = cast_fn(df_clean[col], col)
                df_clean[col] = casted

            except Exception as e:
                print(f"   ❌ Cast failed for '{col}': {e}")

    return df_clean
