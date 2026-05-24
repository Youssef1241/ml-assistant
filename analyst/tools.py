from langchain.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
from dotenv import load_dotenv
from state import prompt_generator
import logging
from logging_utils import get_logger, log_event

load_dotenv()
logger = get_logger(__name__)
@tool
def analyse_data(state: dict) -> str:
    """Analyse the data and return a summary"""
    log_event(logger,logging.INFO,"analyse_data tool start",df_info_keys=list(state.get("df_info", {}).keys()),)
    output_string = prompt_generator(state, ['Sample Data','n_rows', 'n_cols', 'numeric_features', 'categorical_features', 'all_stats','Class Distributions','null_percentages', 'correlation_matrix'])

    os.makedirs("frontend/static", exist_ok=True)
    plt.figure(figsize=(6,6))
    class_dist = state["df_info"]["Class Distributions"]
    class_values = [item["count"] for item in class_dist.values()]
    plt.pie(class_values, labels=class_dist.keys(), autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
    plt.title(f"Target Value Counts")
    plt.savefig("frontend/static/label_pie_chart.png");
    log_event(logger, logging.INFO, "Saved chart", path="frontend/static/label_pie_chart.png")

    plt.figure(figsize=(6,6))
    plt.bar(state["df_info"]["null_counts"].keys(), state["df_info"]["null_counts"].values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Null Values")
    plt.savefig("frontend/static/null_values_bar_chart.png");
    log_event(logger, logging.INFO, "Saved chart", path="frontend/static/null_values_bar_chart.png")

    _,ax=plt.subplots(figsize=(15,15))
    sns.heatmap(pd.DataFrame(state["df_info"]["correlation_matrix"]),cmap='viridis',annot=True,ax=ax,annot_kws={'fontsize':9})
    plt.title('Correlation Matrix for the Dataset')
    plt.savefig("frontend/static/correlation_matrix.png");
    log_event(logger, logging.INFO, "Saved chart", path="frontend/static/correlation_matrix.png")
    log_event(logger, logging.INFO, "analyse_data tool complete", output_chars=len(output_string))
    return output_string

tools = [analyse_data]
tools_by_name = {tool.name: tool for tool in tools}
