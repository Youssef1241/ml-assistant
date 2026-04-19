
import os
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

def update_nulls(state, df):
    impute_columns = state["user_choice"]["null_columns"]["fill_with_average"]
    drop_columns = state["user_choice"]["null_columns"]["drop_column"]
    drop_rows = state["user_choice"]["null_columns"]["drop_rows"]
    impute_response, df = _replace_with_avg(impute_columns, df) if len(impute_columns) > 0 else []
    col_drop_response, df = _drop_column(drop_columns, df) if len(drop_columns) > 0 else []
    row_drop_response, df = _drop_all_rows(drop_rows, df) if len(drop_rows) > 0 else []
    return ([impute_response, col_drop_response, row_drop_response], df)
           
        
def _replace_with_avg(columns: list, df: DataFrame) -> (str, DataFrame):
    """Replace null values with average value"""
    loglist = list()
    total_null_count = 0
    for column in columns:
        try:
            if column not in df.columns:
                loglist.append(f"Column '{column}' does not exist in data.")
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                loglist.append( f"Column '{column}' is not numeric. Cannot replace nulls with average.")
                continue
            avg_val = df[column].mean()
            null_count = df[column].isnull().sum()
            total_null_count+=null_count
            if null_count == 0:
                continue
            df[column] = df[column].fillna(avg_val)
            loglist.append (
                f"Replaced {null_count} null values in column '{column}' with average value {avg_val:.4f}."
            )
        except Exception as e:
            return f"Error processing column '{column}': {e}"
    loglist.append(f"Replaced {total_null_count} null values in columns '{columns}'.")
    return loglist, df;

def _drop_column(columns: list, df: DataFrame) -> (str, DataFrame):
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
        return loglist, df;
    except Exception as e:
        return f"Error dropping columns '{drop_columns}': {e}"

def _drop_all_rows(columns: str, df: DataFrame) -> (str, DataFrame):
    """Drop all rows that contain a null value in the specified column."""
    loglist = list()
    total_rows_removed = 0
    for column in columns:
        try:
            if column not in df.columns:
                loglist.append( f"Column '{column}' does not exist in data.")
            initial_count = df.shape[0]
            null_count = df[column].isnull().sum()
            if null_count == 0:
                loglist.append( f"No rows with null values in column '{column}' were found.")
                continue
            df = df[df[column].notnull()]
            rows_removed = initial_count - df.shape[0]
            total_rows_removed += rows_removed
            loglist.append( f"Dropped {rows_removed} rows with null values in column '{column}'.")
        except Exception as e:
            return f"Error dropping rows for column '{column}': {e}"
    loglist.append( f"Dropped {total_rows_removed} rows with null values in columns '{columns}'.")
    return loglist, df;
