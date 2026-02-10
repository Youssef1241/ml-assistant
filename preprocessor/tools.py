from langchain.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from preprocessor.model import model

@tool
def replace_with_avg(data: str, column_name: str) -> str:
    """Replace null values with average value"""
    try:
        df = pd.read_csv(data)
        if column_name not in df.columns:
            return f"Column '{column_name}' does not exist in data."
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return f"Column '{column_name}' is not numeric. Cannot replace nulls with average."
        avg_val = df[column_name].mean()
        null_count = df[column_name].isnull().sum()
        if null_count == 0:
            return f"No null values present in column '{column_name}'."
        df[column_name].fillna(avg_val, inplace=True)
        df.to_csv(data, index=False)
        return (
            f"Replaced {null_count} null values in column '{column_name}' with average value {avg_val:.4f}."
        )
    except Exception as e:
        return f"Error processing column '{column_name}': {e}"

@tool
def drop_column(data: str, column_name: str) -> str:
    """Drop a column from the dataset."""
    try:
        df = pd.read_csv(data)
        if column_name not in df.columns:
            return f"Column '{column_name}' does not exist in data."
        df.drop(columns=[column_name], inplace=True)
        df.to_csv(data, index=False)
        return f"Column '{column_name}' dropped successfully."
    except Exception as e:
        return f"Error dropping column '{column_name}': {e}"

@tool
def drop_all_rows(data: str, column_name: str) -> str:
    """Drop all rows that contain a null value in the specified column."""
    try:
        df = pd.read_csv(data)
        if column_name not in df.columns:
            return f"Column '{column_name}' does not exist in data."
        initial_count = df.shape[0]
        null_count = df[column_name].isnull().sum()
        if null_count == 0:
            return f"No rows with null values in column '{column_name}' were found."
        df = df[df[column_name].notnull()]
        df.to_csv(data, index=False)
        rows_removed = initial_count - df.shape[0]
        return f"Dropped {rows_removed} rows with null values in column '{column_name}'."
    except Exception as e:
        return f"Error dropping rows for column '{column_name}': {e}"

@tool
def ask_user(question: str) -> str:
    """Prompt the user"""
    return question


tools = [replace_with_avg, drop_column, drop_all_rows, ask_user]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)