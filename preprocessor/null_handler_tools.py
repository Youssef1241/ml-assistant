import os
import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool
from seaborn._core.typing import ContinuousValueSpec
from preprocessor.model import model, struct_model

load_dotenv()

def replace_with_avg(columns: list) -> str:
    """Replace null values with average value"""
    data = os.environ.get("DATA_PATH")
    loglist = list()
    df = pd.read_csv(data)
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
    df.to_csv(data, index=False)
    return loglist;

def drop_column(columns: list) -> str:
    """Drop a column from the dataset."""
    try:
        data = os.environ.get("DATA_PATH")
        loglist = list()
        drop_columns = list()
        df = pd.read_csv(data)
        for column in columns:
            if column not in df.columns:
                loglist.append( f"Column '{column}' does not exist in data.")
                continue
            drop_columns.append(column)
        df.drop(columns=drop_columns, inplace=True)
        df.to_csv(data, index=False)
        loglist.append( f"Columns '{drop_columns}' dropped successfully.")
        return loglist;
    except Exception as e:
        return f"Error dropping columns '{drop_columns}': {e}"

def drop_all_rows(columns: str) -> str:
    """Drop all rows that contain a null value in the specified column."""
    data = os.environ.get("DATA_PATH")
    df = pd.read_csv(data)
    loglist = list()
    total_rows_removed = 0
    for column in columns:
        try:
            if column not in df.columns:
                loglist.append( f"Column '{column}' does not exist in data.")
                ContinuousValueSpec
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
    df.to_csv(data, index=False)
    return loglist;

# tools = [replace_with_avg, drop_column, drop_all_rows]
# tools_by_name = {tool.name: tool for tool in tools}
# model_with_tools = model.bind_tools(tools)