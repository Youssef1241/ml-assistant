from langchain.tools import tool
from analyst.model import model
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json

@tool
def analyse_data(data: str) -> str:
    """Analyse the data and return a summary"""
    output_string =""
    df = pd.read_csv(data)

    output_string += df.head().to_string()
    output_string += "\n\n"
    output_string += df.describe().to_string()

    last_col = df.columns[-1]
    value_counts = df[last_col].value_counts()

    output_string+="\n\n"
    output_string+=value_counts.to_string()

    plt.figure(figsize=(6,6))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
    plt.title(f"{last_col} Value Counts")

    # Save the chart to a file
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/label_pie_chart.png");

    output_string+="\n\nNull Values\n"
    output_string+=df.isnull().sum().to_string()

    plt.figure(figsize=(6,6))
    plt.bar(df.isnull().sum().index, df.isnull().sum().values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Null Values")
    plt.savefig("images/null_values_bar_chart.png");

    numeric_cols = df.select_dtypes(include='number')
    corr_matrix = numeric_cols.corr()
    output_string+="\n\n"
    output_string+=corr_matrix.to_string()

    fig,ax=plt.subplots(figsize=(15,15))
    sns.heatmap(corr_matrix,cmap='viridis',annot=True,ax=ax,annot_kws={'fontsize':9})
    plt.title('Correlation Matrix for the Dataset')

    plt.savefig("images/correlation_matrix.png");

    return output_string



tools = [analyse_data]
tools_by_name = {tool.name: tool for tool in tools}


model_with_tools = model.bind_tools(tools)

