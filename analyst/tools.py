from langchain.tools import tool
from analyst.model import model
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
from dotenv import load_dotenv
from state import prompt_generator

load_dotenv()
@tool
def analyse_data(state: dict) -> str:
    """Analyse the data and return a summary"""
    output_string = prompt_generator(state, ['Sample Data','n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'all_stats','Class Distributions','null_percentages', 'correlation_matrix'])



    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.pie(state["df_info"]["class_dist"].values(), labels=state["df_info"]["class_dist"].keys(), autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
    plt.title(f"Target Value Counts")
    plt.savefig("images/label_pie_chart.png");

    plt.figure(figsize=(6,6))
    plt.bar(state["df_info"]["null_counts"].keys(), state["df_info"]["null_counts"].values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Null Values")
    plt.savefig("images/null_values_bar_chart.png");

    _,ax=plt.subplots(figsize=(15,15))
    sns.heatmap(pd.DataFrame(state["df_info"]["correlation_matrix"]),cmap='viridis',annot=True,ax=ax,annot_kws={'fontsize':9})
    plt.title('Correlation Matrix for the Dataset')
    plt.savefig("images/correlation_matrix.png");
    return output_string

tools = [analyse_data]
tools_by_name = {tool.name: tool for tool in tools}


model_with_tools = model.bind_tools(tools)

