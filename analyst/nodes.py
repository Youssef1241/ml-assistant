from langchain.messages import SystemMessage
from analyst.tools import model_with_tools
import json
import socket


def analyst_call(state: dict):
    """LLM decides whether to call a tool or not"""

    messages_to_send = [
        SystemMessage(
            content="""You are an expert data analyst tasked with analysing csv data for machine learning tasks. Your instructions: 1) call the analyze_data tool immediately and only once with the given address as the data parameter. 2) when it returns,use the analyze_data tool output to analyse the data and return a summary of the data. Your analysis will be in this format:
                        - The analysis will be markdown format, use bold, horizontal lines, and others where appropriate
                        - The analysis will be in the following sections:
                         - Description of the data: number of rows and columns, along with brief description of the data types, pointing out the label column
                         - Discuss data imbalance if there is, showing the labelled image returned from the tool its address is "images/label_pie_chart.png", if the data is not imbalanced, just mention it briefly
                         - Mention the number of nulls in each column, pointing out columns where the nulls are larger than usual, showing the labelled image returned from the tool its address is "images/null_values_bar_chart.png"
                         - Show the correlation matrix image and discuss it, pointing out patterns its address is "images/correlation_matrix.png", if the correlation matrix is not showing any patterns, just mention it briefly
                        """
        )
    ] + state["messages"]

    result = model_with_tools.invoke(messages_to_send)
    return {
        "messages": [result],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

