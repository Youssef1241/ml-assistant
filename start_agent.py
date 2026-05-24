import streamlit as st
from langchain.messages import HumanMessage, SystemMessage
from main import agent
from langgraph.types import Command
from langchain_core.messages import AIMessageChunk
import logging
from logging_utils import get_logger, log_event

logger = get_logger(__name__)

def start_agent(config, file_path=None, data_description=None, target=None):
    log_event(logger,logging.INFO,"start_agent invoked",thread_id=config.get("configurable", {}).get("thread_id"),has_file=bool(file_path),has_description=bool(data_description),target=target,)
    messages = [SystemMessage(content="use markdown, but not ```markdown cell"),HumanMessage(content="Analyze the data")]
    full_text = ""
    if st.session_state.choice == "Azure OpenAI":
        model_info = {
            "api_version": st.session_state.api_version,
            "deployment": st.session_state.deployment,
            "endpoint": st.session_state.endpoint,
            "api_key": st.session_state.key,
            "choice": st.session_state.choice,
        }

    else:
        model_info = {
            "model": st.session_state.model,
            "api_key": st.session_state.key,
            "choice": st.session_state.choice,
        }
    for mode, chunk in agent.stream(
        {"messages": messages, "subgraph": 0, "user_choice": {}, "struct": {}, "df_info": {"target": target, "filepath": file_path, "data_description": data_description}, "pipeline": {}, "data_split": "", "update": {}, "model_info": model_info}, 
        config, 
        stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            # chunk is a tuple (AIMessageChunk, metadata)
            msg, metadata = chunk
            if isinstance(msg, AIMessageChunk) and msg.content:
                # full_text += msg.content
                full_text += msg.content if isinstance(msg.content, str) else ""
                yield msg.content
        elif mode == "updates":
            if "__interrupt__" in chunk:
                st.session_state.interrupt_payload = chunk["__interrupt__"]
                st.session_state.history.append({"role": "assistant", "content": full_text})
                log_event(logger,logging.INFO,"start_agent interrupted",response_chars=len(full_text),interrupt_items=len(chunk["__interrupt__"]),)
                return
    log_event(logger,logging.INFO,"start_agent stream finished without interrupt",response_chars=len(full_text),)

def continue_agent(config, payload):
    log_event(logger,logging.INFO,"continue_agent invoked",thread_id=config.get("configurable", {}).get("thread_id"),payload_type=type(payload).__name__,)
    payload = agent.invoke(Command(resume=payload),config=config)
    print(type(payload))
    print("PAYLOAD: ",payload)
    interrupt_payload = payload["__interrupt__"]
    log_event(logger,logging.INFO,"continue_agent non-stream complete",has_interrupt=bool(interrupt_payload),)
    return interrupt_payload[0].value if interrupt_payload else None

def continue_with_streaming(config, payload):
    full_text = ""
    for mode, chunk in agent.stream(
        Command(resume=payload),
        config, 
        stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
        # chunk is a tuple (AIMessageChunk, metadata)
            msg, metadata = chunk
            if isinstance(msg, AIMessageChunk) and msg.content:
                full_text += msg.content
                yield msg.content
        elif mode == "updates":
            if "__interrupt__" in chunk:
                print("CHUNK IS: ",chunk)
                st.session_state.continue_payload = chunk["__interrupt__"][0].value
                st.session_state.history.append({"role": "assistant", "content": full_text})
                log_event(
                    logger,
                    logging.INFO,
                    "continue_agent stream interrupted",
                    response_chars=len(full_text),
                )
                return
    log_event(
        logger,
        logging.INFO,
        "continue_agent stream finished",
        response_chars=len(full_text),
    )

# def start_agent(config, file_path=None, data_description=None, target=None):
#     return print("Starting agent...")
# def continue_agent(config, payload):
#     return print("Continuing agent...")

def convert_from_snake_case(string):
    if isinstance(string, list):
        return [convert_from_snake_case(item) for item in string]
    else:
        if string.isupper():
            return string
        string = string.replace("_", " ")
        return string.capitalize()
def convert_to_snake_case(string):
    if string.isupper():
        return string
    string = string.replace(" ", "_")
    return string.lower()
def display_list(chosen_list, dict_inside = False):
    final_string = ""
    for item in chosen_list:
        if item == chosen_list[-1]:
            if dict_inside:
                for key, value in item.items():
                    final_string += f"{key.capitalize()}: {value}"
            else:
                final_string += f"{item}"
        else:
            if dict_inside:
                for key, value in item.items():
                    final_string += f"{key.capitalize()}: {value}, "
            else:
                final_string += f"{item}, "
    return final_string
def display_dict(chosen_dict, hp = False):
    final_string = ""
    for key, value in chosen_dict.items():
        if hp:
            final_string += f"{key}: {value}, "
        else:
            final_string += f"{convert_from_snake_case(key)}: {convert_from_snake_case(value)}, "
    if final_string.endswith(", "):
        final_string = final_string[:-2]
    return final_string

