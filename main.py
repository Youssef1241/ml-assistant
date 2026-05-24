import logging
from loguru import logger
from config.model import *
from state import MessagesState
from langgraph.types import interrupt
from langchain.messages import SystemMessage
from logging_utils import get_logger, log_event
from langgraph.graph import StateGraph, START, END
from config.config_dicts.null_handler import null_handler_config

logger = get_logger(__name__)

def build_hitl_subgraph(config: dict, orderlist: list):
    from helpers import create_model_instance
    node_dict= {"analysis": 0, "struct": 0, "interrupt": 0, "update": 0}
    numbered_orderlist = []
    for node in orderlist:
        numbered_orderlist.append(node+str(node_dict[node]))
        node_dict[node]+=1
    def struct_node(state):
        model = create_model_instance(state["model_info"])
        logger.info("Subgraph struct node start")
        node_count = int(numbered_orderlist[state['subgraph']][-1])
        current_struct = config['struct'][node_count] 
        log_event(logger,logging.INFO,"Subgraph struct node start",node_count=node_count,output_name=current_struct.get("output_name"),)
      
        struct_model = model.with_structured_output(current_struct["schema"])

        message_template = current_struct["prompt"]
        context_demands = current_struct.get("context_demands",{})
        if context_demands != {}:
            prompt_string = ""
            for key,value in context_demands.items():
                prompt_string += str(state[key][value]) + "\n\n"
            human_message = message_template.format(context = prompt_string)
        else:
            human_message = message_template
        messages = [SystemMessage(content=prompt_generator(state, current_struct.get("data_demands",[]))), human_message]
        logger.info("Message prompts: " + str(messages))
        try:
            options_dict = struct_model.invoke(messages)
        except Exception as e:
            import traceback

            error_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "status_code": getattr(getattr(e, "response", None), "status_code", None),
                "response_text": getattr(getattr(e, "response", None), "text", None),
                "request_url": getattr(getattr(e, "request", None), "url", None),
            }
            raise RuntimeError(f"struct_model.invoke failed: {error_info}") from e
        # logger.info("Response: " + str(options_dict)) 
        import pickle
        struct_dict = state['struct']

        struct_dict[current_struct["output_name"]] = options_dict.model_dump_json()
        
        return {
            "struct": struct_dict,
            "subgraph": state["subgraph"] + 1,
        }

    def ask_user_node(state):

        if config.get("interrupt_message", None):
            interrupt_message = config["interrupt_message"]
            n_rows = state['df_info']['n_rows']
            n_rows_20 = int(n_rows * 0.2)
            n_rows_10 = int(n_rows * 0.1)
            log_event(
                logger,
                logging.INFO,
                "Subgraph interrupt requested with message",
                n_rows=n_rows,
            )
            user_choice = interrupt({"interrupt_message": interrupt_message.format(n_rows=n_rows, n_rows_20=n_rows_20, n_rows_10=n_rows_10)})
        else:
            log_event(logger, logging.INFO, "Subgraph interrupt requested with struct payload")
            user_choice = interrupt({"struct": state.get("struct",{})})
        state_dict = state['user_choice']
        node_count = int(numbered_orderlist[state['subgraph']][-1])
        state_dict[config["interrupt_name"][node_count]] = user_choice
        
        return {
            "user_choice": {config["interrupt_name"][node_count]: user_choice},
            "subgraph": 0,
        }
              
    graph_builder = StateGraph(MessagesState)
    node_dict = {
        "struct": struct_node,
        "interrupt": ask_user_node,
        "update": config.get("update", {}).get("method", None)
    }

    for i,node in enumerate(orderlist):
        graph_builder.add_node(numbered_orderlist[i],node_dict[node])

    for i,node in enumerate(numbered_orderlist):
        graph_builder.add_edge(START, node) if i == 0 else graph_builder.add_edge(numbered_orderlist[i-1],node)
    graph_builder.add_edge(numbered_orderlist[-1],END)

    graph = graph_builder.compile()
    return graph

from state import *
from processing.llm_call import *
from cleaner.main import clean_data
from analyst.nodes import analyst_call
from processing.main import train_and_test
from processing.main import create_pipeline
from langgraph.graph import StateGraph, START, END
from config.config_dicts.hp_chooser import hp_config
from langgraph.checkpoint.memory import InMemorySaver
from processing.encoding_handling import skew_handling
from cleaner.detection_and_cleaning import make_report
from config.config_dicts.skew_handler import skew_config
from analyst.tools import tools_by_name as analyst_tools
from config.config_dicts.final_chooser import final_config
from config.config_dicts.model_chooser import models_config
from config.config_dicts.filter_chooser import filter_config
from config.config_dicts.metric_chooser import metrics_config
from config.config_dicts.cleaning_handler import clean_config
from config.config_dicts.sampling_query import sampling_config
from config.config_dicts.scaling_handler import scaling_config
from config.config_dicts.null_handler import null_handler_config
from config.config_dicts.encoding_handler import encoding_config
from config.config_dicts.imbalance_handler import imbalance_config
from processing.null_handling import null_handling

agent_builder = StateGraph(MessagesState)
analyst_tool_node = make_tool_node(analyst_tools)

clean_subgraph = build_hitl_subgraph(clean_config, ["struct", 'interrupt'])
null_subgraph = build_hitl_subgraph(null_handler_config, ["struct",'struct','interrupt'])
metric_subgraph = build_hitl_subgraph(metrics_config, ["struct",'interrupt'])
model_subgraph = build_hitl_subgraph(models_config, ["struct",'interrupt'])
encoding_subgraph = build_hitl_subgraph(encoding_config, ["struct",'struct','interrupt'])
skew_subgraph = build_hitl_subgraph(skew_config, ['struct','interrupt'])
scaling_subgraph = build_hitl_subgraph(scaling_config, ['struct','interrupt'])
imbalance_subgraph = build_hitl_subgraph(imbalance_config, ['struct','interrupt'])
filter_subgraph = build_hitl_subgraph(filter_config, ['update', 'struct','interrupt'])
hp_subgraph = build_hitl_subgraph(hp_config, ['struct','struct','interrupt'])
final_subgraph = build_hitl_subgraph(final_config, ['struct','interrupt'])
sampling_subgraph = build_hitl_subgraph(sampling_config, ['interrupt'])

agent_builder.add_node("analyst", analyst_call)
agent_builder.add_node("analyst_tools", analyst_tool_node)
agent_builder.add_node("df info",load_df)
agent_builder.add_node("null", null_subgraph)
agent_builder.add_node("metric", metric_subgraph)
agent_builder.add_node("model", model_subgraph)
agent_builder.add_node("encoding", encoding_subgraph)
agent_builder.add_node("skew", skew_subgraph)
agent_builder.add_node("scaling", scaling_subgraph)
agent_builder.add_node("imbalance", imbalance_subgraph)
agent_builder.add_node("allto6", filter_subgraph)
agent_builder.add_node("hp", hp_subgraph)
agent_builder.add_node("sampling", sampling_subgraph)
agent_builder.add_node("train", train_and_test)
agent_builder.add_node("6to3", final_subgraph)
agent_builder.add_node('eval report', reporter_call)
agent_builder.add_node('interrupt', ask_user)
agent_builder.add_node("create pipeline", create_pipeline)
agent_builder.add_node("final report", report_generator)
agent_builder.add_node("train_full_data", train_and_test)
agent_builder.add_node("null_handling", null_handling)
agent_builder.add_node("skew_handling", skew_handling)
agent_builder.add_node("gather data", make_report)
agent_builder.add_node("clean agent", clean_subgraph)
agent_builder.add_node("clean", clean_data)

agent_builder.add_edge(START, "df info")
agent_builder.add_edge("df info", "analyst")
agent_builder.add_conditional_edges(
    "analyst",
    should_continue,
    ["analyst_tools", "gather data"]
)
agent_builder.add_edge("analyst_tools", "analyst")
agent_builder.add_edge("gather data", "clean agent")
agent_builder.add_edge("clean agent", "clean")

agent_builder.add_conditional_edges(
    "clean",
    reroute_null,
    ["null", "metric"] 
)
agent_builder.add_edge("null", "null_handling")
agent_builder.add_edge("null_handling", "metric")
agent_builder.add_edge("metric", "model")
agent_builder.add_edge("model", "encoding")
agent_builder.add_edge("encoding", "skew")
agent_builder.add_edge("skew", "skew_handling")
agent_builder.add_edge("skew_handling", "skew saver")
agent_builder.add_edge("skew saver","scaling")
agent_builder.add_edge("scaling", "imbalance")
agent_builder.add_edge("imbalance", "allto6")


agent_builder.add_conditional_edges(
    "allto6",
    route_sampling,
    ["sampling", "hp"]
)

agent_builder.add_edge("sampling", "hp")
agent_builder.add_edge("hp", "train")
agent_builder.add_edge("train", "6to3")
agent_builder.add_edge("6to3", "train_full_data")
agent_builder.add_edge("train_full_data", "eval report")
agent_builder.add_edge("eval report", "interrupt")
agent_builder.add_conditional_edges(
    "interrupt",
    reroute_retrain,
    ["null", "metric", "model", "encoding", "skew", "sampling", "scaling", "imbalance", "allto6", "hp", "6to3", "create pipeline"]
)
agent_builder.add_edge("create pipeline", "final report")
agent_builder.add_edge("final report", END)

checkpointer = InMemorySaver()
log_event(logger,logging.INFO,"Compiling LangGraph agent",nodes=["load_df", "analyst", "analyst_tools"],conditional_edge="analyst->(analyst_tools|END)",)
agent = agent_builder.compile(checkpointer=checkpointer)