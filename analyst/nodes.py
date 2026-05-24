from langchain.messages import SystemMessage
from langgraph.graph.state import RunnableConfig
import os
from helpers import create_model_instance
import time
import logging
from dotenv import load_dotenv
import pandas as pd
from helpers import use_persistent
from logging_utils import get_logger, log_event
from analyst.tools import tools
load_dotenv()
logger = get_logger(__name__)

def analyst_call(state: dict, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    log_event(logger,logging.INFO,"Analyst call start",prior_messages=len(state.get("messages", [])),)
    model = create_model_instance(state["model_info"])
    model_with_tools = model.bind_tools(tools)

    messages_to_send = [
        SystemMessage(
            content="""You are an expert data analyst tasked with analysing csv data for machine learning tasks. Your instructions: 1) call the analyze_data tool immediately and only once. 2) when it returns,use the analyze_data tool output to analyse the data and return a summary of the data. Your analysis will be in this format:
                        - The analysis will be markdown format, use bold, horizontal lines, and others where appropriate
                        - The analysis will be in the following sections:
                         - Description of the data: number of rows and columns, along with brief description of the data types, pointing out the label column
                         - Discuss data imbalance if there is, showing the labelled image returned from the tool its address is "app/static/label_pie_chart.png", if the data is not imbalanced, just mention it briefly
                        - Mention the number of nulls in each column, pointing out columns where the nulls are larger than usual, showing the labelled image returned from the tool its address is "app/static/null_values_bar_chart.png"
                         - Show the correlation matrix image and discuss it, pointing out patterns its address is "app/static/correlation_matrix.png", if the correlation matrix is not showing any patterns, just mention it briefly
                        """
        )
    ] + state["messages"]
    started_at = time.perf_counter()
    result = None
    streamed = False
    for chunk in model_with_tools.stream(messages_to_send, config):
        streamed = True
        result = chunk if result is None else result + chunk
    if not streamed:
        result = model_with_tools.invoke(messages_to_send)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log_event(
        logger,
        logging.INFO,
        "Analyst call complete",
        elapsed_ms=elapsed_ms,
        tool_calls=len(getattr(result, "tool_calls", []) or []),
    )
    return {
        "messages": [result],
        "llm_calls": state.get('llm_calls', 0) + 1,
    }