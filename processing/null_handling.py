
import os
import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame
import logging
from logging_utils import get_logger, log_event
from state import load_df, load_df_fromdf

load_dotenv()
logger = get_logger(__name__)

def null_handling(state):
    log_event(logger, logging.INFO, "null_handling start")
    df = pd.read_csv(state["df_info"]["filepath"])
    logs, df = update_nulls(state, df)
    df.to_csv(state["df_info"]["filepath"], index=False)

    state["df_info"].update(load_df_fromdf(df, state['df_info']['target']))

    import pickle
    pickle.dump(logs, open("pickles/null_handling.pkl", "wb"))
    pickle.dump(df.isnull().sum(), open(f"pickles/null_count_after_null_handling.pkl", "wb"))

    return state

def after_null_handling(state):
    import pickle 
    pickle.dump(state, open("pickles/after_null_handling.pkl", "wb"))
    return state

def update_nulls(state, df):
    # Make the saver robust to missing keys in deeply nested dictionaries to avoid KeyError
    user_choice = state.get("user_choice", {})
    null_cols_choice = user_choice.get("null_columns", {}).get("null_columns", {})
    impute_columns = null_cols_choice.get("fill_with_average", [])
    drop_columns = null_cols_choice.get("drop_column", [])
    drop_rows = null_cols_choice.get("drop_rows", [])

    log_event(logger,logging.INFO,"update_nulls start",impute_count=len(impute_columns),drop_columns_count=len(drop_columns),drop_rows_count=len(drop_rows))
    
    impute_response, df = _replace_with_avg(impute_columns, df) if len(impute_columns) > 0 else ([], df)
    col_drop_response, df = _drop_column(drop_columns, df) if len(drop_columns) > 0 else ([], df)
    row_drop_response, df = _drop_all_rows(drop_rows, df) if len(drop_rows) > 0 else ([], df)
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
            log_event(logger, logging.ERROR, "replace_with_avg failed", column=column, error=str(e))
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
        log_event(logger, logging.INFO, "Columns dropped", columns=drop_columns)
        return loglist, df;
    except Exception as e:
        log_event(logger, logging.ERROR, "drop_column failed", columns=drop_columns, error=str(e))
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
            log_event(logger, logging.ERROR, "drop_all_rows failed", column=column, error=str(e))
            return f"Error dropping rows for column '{column}': {e}"
    loglist.append( f"Dropped {total_rows_removed} rows with null values in columns '{columns}'.")
    return loglist, df;
