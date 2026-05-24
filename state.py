import operator
import pandas as pd
from typing import Literal
import logging
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langchain.messages import ToolMessage
from logging_utils import get_logger, log_event
from typing import Any

logger = get_logger(__name__)

def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

def keep_last(left, right):
    return right
    
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    struct: dict
    user_choice: Annotated[dict, merge_dicts]
    update: dict
    subgraph: Annotated[int, keep_last]
    df_info: dict[str,str]
    pipeline: dict
    data_split: Any
    model_info: dict

def should_continue(state: MessagesState) -> Literal["analyst_tools", "null"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        log_event(
            logger,
            logging.INFO,
            "Analyst routing to tools",
            tool_count=len(last_message.tool_calls),
        )
        return "analyst_tools"

    log_event(logger, logging.INFO, "Analyst routing to tools or null")
    return "gather data"

def reroute_retrain(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""    
    log_event(logger, logging.INFO, "rerouting retrain")
    if state['user_choice'].get('retrain', None):
        try:
            state['user_choice']['final'] = None
        except:
            pass
        return state['user_choice'].get('retrain', None)
    else:
        return "create pipeline"

def make_tool_node(tools_by_name):

    def tool_node(state):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            log_event(
                logger,
                logging.INFO,
                "Executing tool call",
                tool_name=tool_call["name"],
                arg_keys=list(tool_call.get("args", {}).keys()),
            )
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

def route_imbalance(state):
    if state["df_info"]["is_imbalanced"]:
        return 'imbalance'
    else:
        return 'allto6'

def route_sampling(state):
    if state["df_info"]["n_rows"] > 2000:
        return 'sampling'
    else:
        return 'hp'

def reroute_null(state):
    null_counts = state['df_info']['null_counts']
    nulls = sum(list(null_counts.values()))
    if nulls > 0:
        return 'null'
    else:
        return 'metric'


def subgraph_reset(state):
    return {
        "subgraph": 0
    }

def prompt_generator(
    state: dict, 
    demands: list[Literal[
        "n_rows",
        "n_cols",
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
        "target_column"
    ]]
):
    df_info = state['df_info']
    output_string = ""
    for item in demands:
        output_string += item + ": \n"
        output_string += str(df_info.get(item, state.get("user_choice").get(item))) + "\n\n"
    if len(demands) == 0:
        output_string = "Perform the task: \n"
    return output_string

def load_df(state):
    def analyze_df(df, target_col):
        from config.config_dicts.imbalance_handler import calculate_imbalance_data
        shape = df.shape
        numeric_features = df.select_dtypes(include="number").columns
        numeric_features = {"count": len(numeric_features),"data": list(numeric_features)}
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        categorical_features = {"count": len(categorical_cols),"data": list(categorical_cols)}
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
                "mean": round(float(df[item].mean()),4),
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
        
        def safe_convert(obj):
            if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
                try:
                    return obj.item()
                except Exception:
                    return obj
            elif isinstance(obj, dict):
                return {safe_convert(k): safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [safe_convert(i) for i in obj]
            else:
                return obj

        serializable_results = {
            "n_rows": int(shape[0]),
            "n_cols": int(shape[1]),
            "numeric_features": safe_convert(numeric_features),
            "categorical_features": safe_convert(categorical_features),
            "numeric_stats": safe_convert(numeric_stats),
            "categorical_stats": safe_convert(cat_stats),
            "null_percentages": safe_convert(null_percentages),
            "Dataset Description": safe_convert(all_stats),
            "Class Distributions": safe_convert(class_percentages),
            "imbalance_ratio": float(ir) if isinstance(ir, float) or hasattr(ir, "item") else ir,
            "correlation_matrix": safe_convert(df[numeric_features["data"]].corr().to_dict()),
            "Sample Data": safe_convert(df.head().to_dict()),
            "Normalized Entropy": float(norm_entropy) if isinstance(norm_entropy, float) or hasattr(norm_entropy, "item") else norm_entropy,
            "is_imbalanced": bool(is_imbalanced),
            "null_counts": safe_convert(null_counts.to_dict() if hasattr(null_counts, "to_dict") else null_counts),
            "target_column": target_col
        }
        return serializable_results
    data_path = state['df_info']['filepath']
    target_col = state['df_info']['target']
    log_event(
        logger,
        logging.INFO,
        "Loading dataset",
        data_path=data_path,
        target_col=target_col,
    )
    df = pd.read_csv(data_path)
    results = analyze_df(df, target_col)
    log_event(
        logger,
        logging.INFO,
        "Dataset analysis complete",
        n_rows=results.get("n_rows"),
        n_cols=results.get("n_cols"),
    )
    state['df_info'].update(results)
    return {
        "df_info": state['df_info'],
    }

def load_df_fromdf(df, target_col):
    def analyze_df(df, target_col):
        from config.config_dicts.imbalance_handler import calculate_imbalance_data
        shape = df.shape
        numeric_features = df.select_dtypes(include="number").columns
        numeric_features = {"count": len(numeric_features),"data": list(numeric_features)}
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        categorical_features = {"count": len(categorical_cols),"data": list(categorical_cols)}
        null_counts = df.isnull().sum()
        null_percentages = {c: round(v/shape[0] *100,2) for c, v in null_counts.items()}
        all_stats = {}
        if target_col in df.columns:
            class_dist = df[target_col].value_counts()
            class_percentages = {}
            for item, cnt in class_dist.items():
                class_percentages[item] = {
                    "count": cnt,
                    "pct": round(cnt / sum(class_dist.values) * 100,2)
                }
        numeric_stats = {}
        cat_stats = {}
        for item in numeric_features["data"]:
            column = df[item].dropna()
            q1, q3 = column.quantile(0.25), column.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((column < lower) | (column > upper)).sum())
            col_dict = {
                "type": "numeric",
                "mean": round(float(df[item].mean()),4),
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
        if target_col in df.columns:
            ir, norm_entropy, is_imbalanced = calculate_imbalance_data(shape[0], class_dist)
        
        def safe_convert(obj):
            if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
                try:
                    return obj.item()
                except Exception:
                    return obj
            elif isinstance(obj, dict):
                return {safe_convert(k): safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [safe_convert(i) for i in obj]
            else:
                return obj

        serializable_results = {
            "n_rows": int(shape[0]),
            "n_cols": int(shape[1]),
            "numeric_features": safe_convert(numeric_features),
            "categorical_features": safe_convert(categorical_features),
            "numeric_stats": safe_convert(numeric_stats),
            "categorical_stats": safe_convert(cat_stats),
            "null_percentages": safe_convert(null_percentages),
            "Dataset Description": safe_convert(all_stats),
            "correlation_matrix": safe_convert(df[numeric_features["data"]].corr().to_dict()),
            "Sample Data": safe_convert(df.head().to_dict()),
            "null_counts": safe_convert(null_counts.to_dict() if hasattr(null_counts, "to_dict") else null_counts),
            "target_column": target_col
        }
        if target_col in df.columns:
            serializable_results["Class Distributions"] = safe_convert(class_percentages)
            serializable_results["Normalized Entropy"] = float(norm_entropy) if isinstance(norm_entropy, float) or hasattr(norm_entropy, "item") else norm_entropy
            serializable_results["is_imbalanced"] = bool(is_imbalanced)
            serializable_results["imbalance_ratio"] = float(ir) if isinstance(ir, float) or hasattr(ir, "item") else ir,
            
        return serializable_results
    log_event(
        logger,
        logging.INFO,
        "Loading dataset",
        target_col=target_col,
    )
    results = analyze_df(df, target_col)
    import pickle
    log_event(
        logger,
        logging.INFO,
        "Dataset analysis complete",
        n_rows=results.get("n_rows"),
        n_cols=results.get("n_cols"),
    )
    return results