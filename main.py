from state import MessagesState
from langgraph.types import interrupt
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage
from config.config_dicts.null_handler import null_handler_config
from config.model import *

def saver(state):
    import pickle
    with open("pickles/skew_results.pkl", "wb") as f:
        pickle.dump(state, f)
    return state

def build_hitl_subgraph(config: dict, orderlist: list):
    node_dict= {"analysis": 0, "struct": 0, "interrupt": 0, "update": 0}
    numbered_orderlist = []
    for node in orderlist:
        numbered_orderlist.append(node+str(node_dict[node]))
        node_dict[node]+=1
    def struct_node(state):
        # import pickle
        # with open("pickles/state.pkl", "rb") as f:
        #     loaded_state = pickle.load(f)

        node_count = int(numbered_orderlist[state['subgraph']][-1])
        current_struct = config['struct'][node_count] 
      
        model = current_struct['model']
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
        # return {"subgraph": 1, "struct": {"structresponse": "value"}}
        options_dict = struct_model.invoke(messages)
        struct_dict = state['struct']
        struct_dict[current_struct["output_name"]] = options_dict.model_dump_json()
        # struct_dict[current_struct["output_name"]] = {"teststrudtnode": "value tsetlsekjl"}

        # try:
        #     import pickle
        #     with open("pickles/struct.pkl", "wb") as f:
        #         pickle.dump(state, f)
        # except Exception as e:
        #     print(f"Error at {e}")
        # return {"struct": options_dict.model_dump_json()}
        
        return {
            "struct": struct_dict,
            "subgraph": state["subgraph"] + 1,
        }
        # else:
        #     return {
        #         "struct": struct_dict,
        #         "subgraph": 1
        #         }
        



    def ask_user_node(state):

        # try:
        #     import pickle
        #     with open("pickles/struct.pkl", "wb") as f:
        #         pickle.dump(state, f)
        # except Exception as e:
        #     print(f"Error at {e}")
        if config.get("interrupt_message", None):
            interrupt_message = config["interrupt_message"]
            n_rows = state['df_info']['n_rows']
            n_rows_20 = int(n_rows * 0.2)
            n_rows_10 = int(n_rows * 0.1)
            user_choice = interrupt({"interrupt_message": interrupt_message.format(n_rows=n_rows, n_rows_20=n_rows_20, n_rows_10=n_rows_10)})
        else:
            user_choice = interrupt({"struct": state.get("struct",{})})
        state_dict = state['user_choice']
        node_count = int(numbered_orderlist[state['subgraph']][-1])
        state_dict[config["interrupt_name"][node_count]] = user_choice
        return {
            "user_choice": state_dict,
            "subgraph": state["subgraph"] + 1,
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

from analyst.nodes import analyst_call
# from config.config_dicts.null_handler import null_handler_config
# from config.config_dicts.metric_chooser import metrics_config
# from config.config_dicts.model_chooser import models_config
from config.config_dicts.encoding_handler import encoding_config
from config.config_dicts.skew_handler import skew_config
# from config.nodes import preprocessor_call
from state import *
from langgraph.graph import StateGraph, START, END
from analyst.tools import tools_by_name as analyst_tools
from langgraph.checkpoint.memory import InMemorySaver

agent_builder = StateGraph(MessagesState)
analyst_tool_node = make_tool_node(analyst_tools)
null_subgraph = build_hitl_subgraph(skew_config, ['struct','struct','interrupt'])
# agent_builder.add_node("analyst", analyst_call)
# agent_builder.add_node("analyst_tools", analyst_tool_node)

agent_builder.add_node("load_df",load_df)
agent_builder.add_node("null_subgraph", null_subgraph)
agent_builder.add_node("saver", saver)
# Add edges to connect nodes
# TEMPORARY
agent_builder.add_edge(START, "load_df")
agent_builder.add_edge("load_df", "null_subgraph")
# agent_builder.add_conditional_edges(
#     "analyst",
#     should_continue,
#     ["analyst_tools", "saver"]
# )
agent_builder.add_edge("null_subgraph", "saver")
agent_builder.add_edge("saver", END)

checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "chat-1",
    }
}


from langchain.messages import HumanMessage, SystemMessage
messages = [SystemMessage(content="start your task"),HumanMessage(content="Analyze the data")]

# Use stream() instead of invoke() to enable input() functionality
for chunk in agent.stream({"messages": messages, "subgraph": 0, "user_choice": {}, "struct": {}, "df_info": {"target": "Survived", "filepath": "datasets/Titanic-Dataset.csv", "data_description": "Test"}}, config): 
    # Process each chunk as it arrives
    for node_name, node_output in chunk.items():
        print(f"\n[{node_name}]")
        if node_name == "__interrupt__":
            interrupt_payload = node_output
            break
        if "messages" in node_output and len(node_output["messages"]) > 0:
            last_message = node_output["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print(f"Content: {last_message.content}")
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"Tool calls: {last_message.tool_calls}")
        else:
            print(f"Output: {node_output}")