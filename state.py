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
    update: Annotated[list[dict], operator.add]
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
    def get_chunk_dims(filepath, max_cells=100_000):
        peek = pd.read_csv(filepath, nrows=0)
        n_cols = len(peek.columns)
        cols = peek.columns.tolist()
        rows_per_chunk = max_cells // n_cols
        cols_per_chunk = max_cells // rows_per_chunk if rows_per_chunk > 0 else n_cols
        col_groups = [
            cols[i: i + cols_per_chunk]
            for i in range(0,n_cols, cols_per_chunk)
        ]
        return rows_per_chunk, col_groups

    def analyze_chunked(filepath, target_col, max_cells = 100_000):
        from preprocessor.config.imbalance_handler import calculate_imbalance_data
        from collections import defaultdict
        from river import stats
        
        rows_per_chunk, col_groups = get_chunk_dims(filepath, max_cells)
        col_groups = [
            list(set(group + [target_col]))
            for group in col_groups
        ]
        all_stats = {}
        null_counts = {}
        class_dist = defaultdict(int)
        corr_list = []
        for col_group in col_groups:
            corr_chunks = []
            n_rows = 0
            variances_dict = defaultdict(stats.Var)
            skew_dict = defaultdict(stats.Skew)
            group_numeric_stats = defaultdict(list)
            group_null_counts = None
            group_cat_counts = defaultdict(lambda: defaultdict(int))
            head_list = []
            for chunk in pd.read_csv(filepath, usecols=col_group, chunksize = rows_per_chunk):
                n_rows += len(chunk)
                sample_chunk = chunk.sample(frac=0.1)
                if len(sample_chunk) == 0:
                    sample_chunk = chunk.sample(n=1)
                corr_chunks.append(sample_chunk.select_dtypes(include=["number"]))

                if group_null_counts is None:
                    group_null_counts = chunk.isnull().sum()
                    head_list.append(chunk.head())
                else:
                    group_null_counts += chunk.isnull().sum()

                if target_col in chunk.columns and col_group == col_groups[0]:
                    for val, cnt in chunk[target_col].value_counts().items():
                        class_dist[val] += cnt

                for col in chunk.select_dtypes(include="number").columns:
                    if col == target_col:
                        continue

                    group_numeric_stats[col].append({
                        "mean": chunk[col].mean(),
                        "min": chunk[col].min(),
                        "max": chunk[col].max(),
                        "n": len(chunk[col])
                    })
                    
                    clean_chunk = chunk[col].dropna()
                    for value in clean_chunk:
                        variances_dict[col].update(value)
                        skew_dict[col].update(value)

                for col in chunk.select_dtypes(include=["object","string"]).columns:
                    if col == target_col:
                        continue
                    for val, cnt in chunk[col].value_counts().head(10).items():
                        group_cat_counts[col][val] += cnt
            corr_df = pd.concat(corr_chunks, axis=0, ignore_index=True)
            corr_list.append(corr_df)
            for col, stats in group_numeric_stats.items():
                total_n = sum(s["n"] for s in stats)
                all_stats[col] = {
                    "type": "numeric",
                    "mean": round(sum(s["mean"] * s["n"] for s in stats)/total_n,4),
                    "std": variances_dict[col].get() **0.5,
                    "min": min(s["min"] for s in stats),
                    "max": max(s["max"] for s in stats),
                    "skew": skew_dict[col].get()
                }

            for col, counts in group_cat_counts.items():
                all_stats[col] = {
                    "type": "categorical",
                    "cardinality": len(counts),
                    "top_values": dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3])
                }
            if group_null_counts is not None:
                null_counts.update(group_null_counts.to_dict())
        class_dist = dict(class_dist)
        corr_list = pd.concat(corr_list, axis=1)
        corr_matrix = corr_list.corr()
        corr_matrix = corr_matrix.dropna(axis=1, how='all')
        corr_matrix = corr_matrix.dropna(axis=0, how='all')
        head = pd.concat(head_list, axis=1)
        ir, norm_entropy, is_imbalanced = calculate_imbalance_data(len(class_dist), n_rows, class_dist)
        per_class = {
        cls: {
            "count": cnt,
            "pct": round(cnt / sum(class_dist.values()) * 100,2)
        }
        for cls, cnt in class_dist.items()
        }
        cat_stats = [
            {"name": key,**value} 
            for key, value in all_stats.items()
            if value["type"] == "categorical"
        ]
        numeric_stats = [
            {"name": key,**value} 
            for key, value in all_stats.items()
            if value["type"] == "numeric"
        ]
        numeric_cols = [stat["name"] for stat in numeric_stats]
        for i,col in enumerate(numeric_cols):
            df = pd.read_csv(filepath, usecols=[col])
            df = df.dropna()
            q1, q3 = df.quantile(0.25), df.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((df < lower) | (df > upper)).sum())
            numeric_stats[i]["outlier_count"] = outlier_count

        return {
                "n_rows": n_rows,
                "n_cols (without target)": sum(len(g) for g in col_groups) - len(col_groups),
                "numeric_features": [c for c, s in all_stats.items() if s["type"] == "numeric"],
                "categorical_features": [c for c, s in all_stats.items() if s["type"] == "categorical"],
                "null_counts": null_counts,
                "null_percentages": {c: round(v/n_rows *100,2) for c, v in null_counts.items()},
                "Dataset Description": all_stats,
                "Class Distributions": per_class,
                "imbalance_ratio": ir,
                "correlation_matrix": corr_matrix.to_dict(),
                "Sample Data": head.to_dict(),
                "Normalized Entropy": norm_entropy,
                "is_imbalanced": is_imbalanced,
                "categorical_stats": cat_stats,
                "numeric_stats": numeric_stats,
            }
    data_path = state['df_info']['filepath']
    target_col = state['df_info']['target']
    results = analyze_chunked(data_path, target_col)
    state['df_info'].update(results)
    return {
        "df_info": state['df_info'],
    }