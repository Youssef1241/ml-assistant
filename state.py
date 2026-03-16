import operator
import pandas as pd
from typing import Literal
from langgraph.graph import END
from langgraph.types import interrupt
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langchain.messages import ToolMessage, HumanMessage
import os


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    struct: dict
    analysis: Annotated[list[dict], operator.add]
    user_choice: dict
    update: dict
    subgraph: int
    original_df: dict
    df_dict: dict[str, pd.DataFrame]
    df_info: dict[str,str]

def preprocessor_should_continue(state: MessagesState) -> Literal["preprocessor_tools","user_input", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        name = last_message.tool_calls[0]["name"]
        if name == "ask_user":
            return "user_input"
        else:
            return "preprocessor_tools"

    # If the LLM asked a question directly (no tool call), still route to user input.
    if getattr(last_message, "content", "") and "?" in last_message.content:
        return "user_input"

    return END

def should_continue(state: MessagesState) -> Literal["analyst_tools", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "analyst_tools"

    return "preprocessor"

def make_tool_node(tools_by_name):

    def tool_node(state):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            tools_args = {**tool_call["args"], "state": state}
            observation = tool.invoke(tools_args)

            result.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": result}

    return tool_node

def ask_user_node(state):
    last = state["messages"][-1]
    if last.tool_calls:
        question = last.tool_calls[0]["args"].get("question", "Provide input:")
    else:
        # Fallback when the model asked directly in assistant content.
        question = last.content or "Provide input:"
    print("\nLLM: ", question)
    
    user_answer = interrupt("Your response: ")
    return {
        "messages": [HumanMessage(content=user_answer)]
    }

def route_imbalance(state):
    if state["df_info"]["is_imbalanced"]:
        return 'imbalance_subgraph'
    else:
        return 'metric_subgraph'

def subgraph_reset(state):
    return {
        "subgraph": 0
    }

def prompt_generator(
    state: dict, 
    demands: list[Literal[
        "n_rows",
        "n_cols (without target)",
        "numeric_features",
        "categorical_features",
        "null_counts",
        "null_percentages",
        "Dataset Description",
        "Class Distributions",
        "imbalance_ratio",
        "correlation_matrix",
        "Sample Data",
        "Normalized Entropy",
        "is_imbalanced",
        "data_description",
        "metrics",
        "models",
        "encoding",
        "imbalance_methods",
        "null_columns",
        "skew",
        "scaling",
    ]]
):
    df_info = state['df_info']
    output_string = ""
    for item in demands:
        output_string += item + ": \n"
        output_string += str(df_info.get(item, state["user_choice"][item])) + "\n\n"
    if len(demands) == 0:
        output_string = "Perform the task: \n"
    return output_string

def load_df(state):
    def analyze_df(df, target_col):
        from config.config_dicts.imbalance_handler import calculate_imbalance_data
        shape = df.shape
        numeric_features = df.select_dtypes(include="number").columns
        numeric_features = {"count": len(numeric_features),"data": numeric_features}
        categorical_features = df.select_dtypes(include=["object","string"]).columns
        categorical_features = {"count": len(categorical_features),"data": categorical_features}
        null_counts = df.isnull().sum()
        null_percentages = {c: round(v/shape[0] *100,2) for c, v in null_counts.items()}
        all_stats = {}
        class_dist = df[target_col].value_counts()
        class_percentages = {}
        numeric_stats = {}
        cat_stats = {}
        for item, cnt in class_dist.items():
            class_percentages[item] = {
                "count": cnt,
                "pct": round(cnt / sum(class_dist.values) * 100,2)
            }
        for item in numeric_features["data"]:
            column = df[item].dropna()
            q1, q3 = column.quantile(0.25), column.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((column < lower) | (column > upper)).sum())
            col_dict = {
                "type": "numeric",
                "mean": round(df[item].mean(),4),
                "std": float(df[item].std()),
                "min": float(df[item].min()),
                "max": float(df[item].max()),
                "skew": float(df[item].skew()),
                "outlier_count": outlier_count
            }
            all_stats[item] = col_dict
            numeric_stats[item] = col_dict

        for item in categorical_features["data"]:
            col_dict = {
                "type": "categorical",
                "cardinality": df[item].nunique(),
                "top_values": dict(sorted(df[item].value_counts().items(), key=lambda x: x[1], reverse=True)[:5])
            }
            all_stats[item] = col_dict
            cat_stats[item] = col_dict
        ir, norm_entropy, is_imbalanced = calculate_imbalance_data(shape[0], class_dist)
        

        return {
            "n_rows": shape[0],
            "n_cols": shape[1],
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_stats": numeric_stats,
            "categorical_stats": cat_stats,
            "null_percentages": null_percentages,
            "Dataset Description": all_stats,
            "Class Distributions": class_percentages,
            "imbalance_ratio": ir,
            "correlation_matrix": df[numeric_features["data"]].corr().to_dict(),
            "Sample Data": df.head().to_dict(),
            "Normalized Entropy": norm_entropy,
            "is_imbalanced": is_imbalanced,
        }
    data_path = state['df_info']['filepath']
    target_col = state['df_info']['target']
    df = pd.read_csv(data_path)
    results = analyze_df(df, target_col)
    state['df_info'].update(results)
    return {
        "df_info": state['df_info'],
    }