from hmac import new
import os
import uuid
import json
import logging
import tempfile
import streamlit as st
from start_agent import *
from collections import defaultdict
from logging_utils import get_logger, log_event, log_values_with_types
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(layout="wide")
left_pad, center_container, right_pad = st.columns([1, 4, 1])
st.set_page_config(page_title="ML Assistant", page_icon="🤖")
with center_container:
    st.title("ML Assistant")
    st.markdown("Welcome to the ML Assistant! This tool will guide you through the machine learning pipeline")
    logger = get_logger(__name__)

    if "step" not in st.session_state:
        st.session_state.step = 0
        st.session_state.history = []
        log_event(logger, logging.INFO, "Initialized session state", step=0)

    if "messages_order" not in st.session_state:
        with open("frontend/messages.json", "r") as f:
            st.session_state.messages_order = json.load(f)

    with st.container(border=False,height=700):
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        current_message = st.session_state.messages_order[st.session_state.step]
        if st.session_state.step == len(st.session_state.messages_order)-1:
            next_message = st.session_state.messages_order[st.session_state.step]
        else:
            next_message = st.session_state.messages_order[st.session_state.step + 1]
        message_type = current_message["type"]
        response_tag = current_message.get("response_tag", None)
        log_event(logger,logging.INFO,"Rendering message step",step=st.session_state.step,message_type=message_type,response_tag=response_tag,)
        if message_type == "file":
            with st.chat_message("assistant"):
                question = "Please upload your dataset (in CSV format) to continue."
                st.write(question)
            
            uploaded_file = st.file_uploader("Choose a file")
            
            if uploaded_file is not None and st.button("Confirm"):
                temp_dir = tempfile.mkdtemp() 
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.data_info = {"file_path": file_path}
                log_event(logger,logging.INFO,"Dataset uploaded",filename=uploaded_file.name,path=file_path,)
                st.session_state.history.append({"role": "assistant", "content": question})
                st.session_state.history.append({"role": "user", "content": f"Uploaded: {uploaded_file.name}"})
                st.session_state.step += 1
                st.rerun()

        elif message_type == "free":
            # st.session_state.step += 1
            # st.session_state.data_info["data_description"] = ""
            # st.rerun()
            if prompt := st.chat_input(current_message["prompt"]):
                if current_message["name"] == "data_description":
                    st.session_state.data_info["data_description"] = prompt
                st.session_state.history.append({"role": "assistant", "content": current_message["prompt"]})
                st.session_state.history.append({"role": "user", "content": prompt})
                st.session_state.step += 1
                st.rerun()

        elif message_type == "target":
            if "start_report" not in st.session_state:
                st.session_state.start_report = None
            question = current_message["prompt"]
            log_values_with_types(logger,logging.INFO,"target step current_message fields",question=question,)
            import pandas as pd
            df = pd.read_csv(st.session_state.data_info["file_path"])
            columns = list(df.columns)
            with st.chat_message("assistant"):
                st.markdown(question)
            selected = st.radio(label="Choose the label column", options=columns, horizontal=True)
            if st.button("Confirm"):
                st.session_state.history.append({"role": "assistant", "content": question})
                st.session_state.history.append({"role": "user", "content": "Target Column: " + selected})
                st.session_state.data_info["target"] = selected
                st.session_state.session_id = str(uuid.uuid4())
                log_event(logger,logging.INFO,"Target selected, starting agent",target=selected,session_id=st.session_state.session_id,)
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                start_chunks = []
                def capture_start_agent():
                    for chunk in start_agent(config, **st.session_state.data_info):
                        start_chunks.append(chunk)
                        yield chunk
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing the data..."):
                        st.write_stream(capture_start_agent())
                if type(start_chunks) == str:
                    st.session_state.start_report = start_chunks
                else:
                    texts = []
                    for chunk in start_chunks:
                        if isinstance(chunk, str):
                            texts.append(chunk)
                        elif isinstance(chunk, dict):
                            # Gemini returns AIMessageChunk as dict — extract content
                            texts.append(chunk.get("content", "") or chunk.get("text", "") or "")
                        elif isinstance(chunk, list):
                            for item in chunk:
                                if isinstance(item, str):
                                    texts.append(item)
                                elif isinstance(item, dict):
                                    texts.append(item.get("content", "") or item.get("text", "") or "")
                    st.session_state.start_report = "".join(texts) 
                nulls = df.isnull().sum().sum()
                st.session_state.nulls = nulls
                rows = df.shape[0]
                st.session_state.rows = rows
                if next_message["type"] == "multi-choice":
                    next_message["options"] = json.loads(st.session_state.interrupt_payload[0].value["struct"][next_message["node_name"][0]])
                    next_message["prompts"] = json.loads(st.session_state.interrupt_payload[0].value["struct"][next_message["node_name"][1]])
                elif next_message["type"] == "choice":
                    next_message["options"] = st.session_state.interrupt_payload[0].value["interrupt_message"]
                else:
                    next_message["options"] = json.loads(st.session_state.interrupt_payload[0].value["struct"][next_message["node_name"]])

                st.session_state.step += 1
                st.rerun()

        elif message_type == "checkbox" or (message_type == "multi-choice" and response_tag == "filter"):
            if response_tag == "scaling":
                question = current_message["options"]["prompts"][0]
                options = current_message["options"]
                log_values_with_types(logger,logging.INFO,"checkbox scaling current_message fields",question=question,options=options,)
                with st.chat_message("assistant"):
                    st.markdown("**" + question + "**")
                all_options = [i for sublist in options["actions"].values() for i in sublist]
                selected_options = defaultdict(list)
                for item in all_options:
                    st.markdown(convert_from_snake_case(item))
                    for action in options["actions"].keys():
                        selected = st.checkbox(label=convert_from_snake_case(action), value=bool(item in options["actions"][action]), key=str(item)+str(action))
                        if selected: selected_options[action].append(item)
                if st.button("Confirm"):
                    st.session_state.history.append({"role": "assistant", "content": question})
                    st.session_state.history.append({"role": "user", "content": "Scaling selections: " + display_dict(selected_options)})
                    log_event(logger,logging.INFO,"Scaling selections submitted",selected_groups=len(selected_options),)
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    response_payload = {response_tag: selected_options}
                    log_values_with_types(logger,logging.INFO,"scaling continue_agent request",response_payload=response_payload,)
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            interrupt_payload = continue_agent(config, response_payload)
                    log_values_with_types(logger,logging.INFO,"scaling continue_agent interrupt",interrupt_payload=interrupt_payload,)
                    if next_message["type"] == "multi-choice":
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                        next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                    elif next_message["type"] == "choice":
                        next_message["options"] = interrupt_payload["interrupt_message"]
                    else:
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                    st.session_state.step += 1
                    st.rerun()
            elif response_tag == "filter":
                options = current_message["options"]
                fil1 = json.loads(current_message["prompts"])
                prompt = fil1["prompts"][0]
                actions = fil1["actions"]
                
                log_values_with_types(logger,logging.INFO,"checkbox filter current_message fields",question=prompt,options=options,prompts=current_message["prompts"],)
                options_comb = []
                recommended_options = list(actions.values())
                options_comb = [f"{convert_from_snake_case(comb["scaling"])}, {convert_from_snake_case(comb["model"])}, {convert_from_snake_case(comb["imbalance_method"])}" for comb in options]
                recommended_comb = [f"{convert_from_snake_case(comb["scaling"])}, {convert_from_snake_case(comb["model"])}, {convert_from_snake_case(comb["imbalance_method"])}" for comb in recommended_options]
                with st.chat_message("assistant"):
                    st.markdown(prompt)
                selected_options = []
                for i,item in enumerate(options_comb):
                    selected = st.checkbox(label=item, value=bool(item in recommended_comb), key=item)
                    if selected: selected_options.append(options[i])
                if st.button("Confirm"):
                    st.session_state.history.append({"role": "assistant", "content": prompt})
                    st.session_state.history.append({"role": "user", "content": "Filter selections: " + display_list(selected_options, dict_inside = True)})
                    log_event(logger,logging.INFO,"Filter selections submitted",selected_count=len(selected_options),)
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    response_payload = {response_tag: selected_options}
                    log_values_with_types(logger,logging.INFO,"filter continue_agent request",response_payload=response_payload,)
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            interrupt_payload = continue_agent(config, response_payload)
                            # import pickle
                            # interrupt_payload = pickle.load(open("pickles/interrupt_payload_hp.pkl", "rb"))
                    log_values_with_types(logger,logging.INFO,"filter continue_agent interrupt",interrupt_payload=interrupt_payload,)
                    if st.session_state.rows <= 2000:
                        st.session_state.step += 1
                    next_message = st.session_state.messages_order[st.session_state.step + 1]
                    if next_message["type"] == "multi-choice" or next_message["type"] == "hp":
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                        next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                    elif next_message["type"] == "choice":
                        next_message["options"] = interrupt_payload["interrupt_message"]
                    else:
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                    st.session_state.step += 1
                    st.rerun()
            elif response_tag == "final":
                if "report_final" not in st.session_state:
                    st.session_state.report_final = None
                question = current_message["options"]["prompts"][0]
                options = current_message["options"]
                log_values_with_types(logger,logging.INFO,"checkbox final current_message fields",question=question,options=options,)
                with st.chat_message("assistant"):
                    st.markdown(question)
                all_options = [i for sublist in options["actions"].values() for i in sublist]
                selected_options = {}
                for item in all_options:
                    display_item = display_list(item.split("-"))
                    selected_options[item]= st.checkbox(label=display_item, value=bool(item in options["actions"]["True"]), key=item)
                if st.button("Confirm"):
                    st.session_state.history.append({"role": "assistant", "content": question})
                    st.session_state.history.append({"role": "user", "content": "Chosen items: " + display_list([item for item,value in selected_options.items() if value == True])})
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    selected_items = [item for item in selected_options.keys() if selected_options[item]]
                    response_payload = {response_tag: {"True": selected_items, "retrain": "full-retrain"}}
                    log_values_with_types(logger,logging.INFO,"final continue_with_streaming request",response_payload=response_payload,)
                    with st.spinner("Generating Report..."):
                        streamed_chunks = []
                        def capture_final_stream():
                            for chunk in continue_with_streaming(config, response_payload):
                                streamed_chunks.append(chunk)
                                yield chunk
                        with st.chat_message("assistant"):
                            st.write_stream(capture_final_stream())
                        st.session_state.report_final = "".join(streamed_chunks)
                    
                    if next_message["type"] == "multi-choice":
                        next_message["options"] = st.session_state.continue_payload["struct"][next_message["node_name"][0]]
                        next_message["prompts"] = st.session_state.continue_payload["struct"][next_message["node_name"][1]]
                    else:
                        next_message["options"] = st.session_state.continue_payload["struct"]
                    log_event(logger, logging.INFO, "Final options confirmed", selected_count=len(selected_items))
                    st.session_state.step += 1
                    st.rerun()
            elif response_tag == "skew":
                question = current_message["options"]["prompts"][0]
                options = current_message["options"]
                log_values_with_types(logger,logging.INFO,"checkbox skew current_message fields",question=question,options=options,)
                with st.chat_message("assistant"):
                    st.markdown(question)
                all_options = [i for sublist in options["actions"].values() for i in sublist]
                selected_options = {}
                for item in all_options:
                    selected_options[item]= st.checkbox(label=item, value=bool(item in options["actions"]["skewed"]), key=item)
                if st.button("Confirm"):
                    print(selected_options)
                    st.session_state.history.append({"role": "assistant", "content": question})
                    st.session_state.history.append({"role": "user", "content": "Columns  selected: " + display_list([item for item,value in selected_options.items() if value == True])}) 
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    selected_items = [item for item in selected_options.keys() if selected_options[item]]
                    response_payload = {response_tag: selected_items}
                    log_values_with_types(logger,logging.INFO,"skew continue_agent request",response_payload=response_payload,)
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            interrupt_payload = continue_agent(config, response_payload)
                    log_values_with_types(logger,logging.INFO,"skew continue_agent interrupt",interrupt_payload=interrupt_payload,)
                    if next_message["type"] == "multi-choice":
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                        next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                    elif next_message["type"] == "choice":
                        next_message["options"] = interrupt_payload["interrupt_message"]
                    else:
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                    log_event(logger, logging.INFO, "Checkbox options confirmed", selected_count=len(selected_items))
                    st.session_state.step += 1
                    st.rerun()

            else:
                question = current_message["options"]["prompts"][0]
                options = current_message["options"]
                log_values_with_types(logger,logging.INFO,"checkbox default current_message fields",question=question,options=options,)
                with st.chat_message("assistant"):
                    st.markdown(question)
                all_options = [i for sublist in options["actions"].values() for i in sublist]
                selected_options = {}
                for item in all_options:
                    selected_options[item] = st.checkbox(label=convert_from_snake_case(item), value=bool(item in options["actions"]["True"]), key=item)
                if st.button("Confirm"):
                    print("selected_options: ",selected_options)
                    st.session_state.history.append({"role": "assistant", "content": question})
                    st.session_state.history.append({"role": "user", "content": "Chosen items: " + display_list([convert_from_snake_case(item) for item,value in selected_options.items() if value == True])})
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    selected_items = [item for item in selected_options.keys() if selected_options[item]]
                    response_payload = {response_tag: {"True": selected_items}}
                    log_values_with_types(logger,logging.INFO,"checkbox default continue_agent request",response_payload=response_payload,)
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            interrupt_payload = continue_agent(config, response_payload)
                    log_values_with_types(logger,logging.INFO,"checkbox default continue_agent interrupt",interrupt_payload=interrupt_payload,)
                    if next_message["type"] == "multi-choice":
                        if next_message["response_tag"] == "filter":
                            next_message["options"] = interrupt_payload["struct"][next_message["node_name"][0]]
                            next_message["prompts"] = interrupt_payload["struct"][next_message["node_name"][1]]
                        else:
                            next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                            next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                    else:
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                    log_event(logger, logging.INFO, "Checkbox options confirmed", selected_count=len(selected_items))
                    st.session_state.step += 1
                    st.rerun()
        elif message_type == "multi-choice":
            if "outer_index" not in st.session_state:
                st.session_state.outer_index = 0
                st.session_state.selected_items = defaultdict(list)
            if response_tag == "encoding" and st.session_state.outer_index == 0:
                st.session_state.selected_items = defaultdict(list)
            
            log_values_with_types(logger,logging.INFO,"multi-choice step state",outer_index=st.session_state.outer_index,options=current_message["options"],prompts=current_message["prompts"],)
            options_dict = current_message["options"]["actions"]
            prompts = current_message["prompts"]["prompts"]
            outer_list = []
            filtered_options_dict = {}
            filtered_prompts = []
            for i, item in enumerate(options_dict.keys()):
                if options_dict[item] != []:
                    filtered_options_dict[item] = options_dict[item]
                    filtered_prompts.append(prompts[i])
            outer_list = list(filtered_options_dict.keys())
            capitals_outer_list = [convert_from_snake_case(item) for item in outer_list]
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    st.markdown(filtered_prompts[st.session_state.outer_index])
            current_selection = {}
            for col in filtered_options_dict[outer_list[st.session_state.outer_index]]:
                current_selection[col] = st.radio(label = col,options=capitals_outer_list, index=st.session_state.outer_index)
            confirm_button = st.button("Confirm")
            if confirm_button and st.session_state.outer_index < len(outer_list) - 1:
                for col, action in current_selection.items():
                    action = convert_to_snake_case(action)
                    st.session_state.selected_items[action].append(col)
                st.session_state.history.append({"role": "assistant", "content": filtered_prompts[st.session_state.outer_index]})
                st.session_state.history.append({"role": "user", "content": f"Action Chosen for {display_list(list(current_selection.keys()))}: {current_selection[col]}"})
                st.session_state.outer_index += 1
                log_event(logger,logging.INFO,"Multi-choice intermediate selection",current_index=st.session_state.outer_index,)
                print(current_selection)
                st.rerun()
            elif confirm_button:
                for col, action in current_selection.items():
                    action = convert_to_snake_case(action)
                    st.session_state.selected_items[action].append(col)
                st.session_state.history.append({"role": "assistant", "content": filtered_prompts[st.session_state.outer_index]})
                st.session_state.history.append({"role": "user", "content": f"Action Chosen for {display_list(list(current_selection.keys()))}: {current_selection[col]}"})
                st.session_state.outer_index = 0
                st.session_state.selected_items = dict(st.session_state.selected_items)
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                log_event(logger, logging.INFO, "Submitting multi-choice selections")
                response_payload = {response_tag: st.session_state.selected_items}
                log_values_with_types(logger,logging.INFO,"multi-choice continue_agent request",response_payload=response_payload)
                with st.chat_message("assistant"):
                    with st.spinner("Handling Null Values..."):
                        interrupt_payload = continue_agent(config, response_payload)
                log_values_with_types(logger,logging.INFO,"multi-choice continue_agent interrupt",interrupt_payload=interrupt_payload)
                if next_message["type"] == "multi-choice":
                    next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                    next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                elif next_message["type"] == "choice":
                    next_message["options"] = interrupt_payload["interrupt_message"]
                elif next_message["type"] == "model-choice":
                    next_message["options"] = interrupt_payload["struct"]
                else:
                    next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                st.session_state.step += 1
                st.rerun()
        elif message_type == "choice":
            log_values_with_types(logger,logging.INFO,"choice step current_message fields",prompt=current_message["options"],)
            with st.chat_message("assistant"):
                st.markdown(current_message["options"])
            # cols = st.columns(2)
            options = ["Yes", "No"]
            for i, col in enumerate(options):
                if st.button(options[i], key=f"btn_{i}"):
                    st.session_state.history.append({"role": "assistant", "content": current_message["options"]})
                    st.session_state.history.append({"role": "user", "content": options[i]})
                    log_event(logger, logging.INFO, "Choice selected", choice=options[i])
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    if options[i] == "Yes":
                        response_payload = {response_tag: True}
                    else:
                        response_payload = {response_tag: False}
                    log_values_with_types(logger,logging.INFO,"choice continue_agent request",response_payload=response_payload,)
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            interrupt_payload = continue_agent(config, response_payload)
                    log_values_with_types(logger,logging.INFO,"choice continue_agent interrupt",interrupt_payload=interrupt_payload,)
                    if next_message["type"] == "multi-choice" or next_message["type"] == "hp":
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                        next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                    elif next_message["type"] == "choice":
                        next_message["options"] = interrupt_payload["interrupt_message"]
                    else:
                        print('look for scaling sca0', interrupt_payload['struct'])
                        next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                    st.session_state.step += 1
                    st.rerun()
        elif message_type == "hp":
            hp0 = current_message["options"]["actions"]
            hp1 = current_message["prompts"]["prompts"]
            log_values_with_types(logger,logging.INFO,"hp step current_message fields",options=current_message["options"],prompts=current_message["prompts"],)
            if "hp_index" not in st.session_state:
                st.session_state.hp_index = 0
                st.session_state.selected_hps = {}

            current_message["models"] = list(hp0.keys())
            with st.chat_message("assistant"):
                st.markdown(hp1[st.session_state.hp_index])
            selected_items = {model: {} for model in current_message["models"]}
            for name, param in current_message["hyperparameters"][current_message["models"][st.session_state.hp_index]].items():
                default = hp0[current_message["models"][st.session_state.hp_index]][name]
                if type(param[0]) == float:
                    default = float(default)
                elif type(param[0]) == int:
                    default = int(default)
                elif type(param[0]) == str:
                    default = str(default)
                if type(param[0]) == float or type(param[0]) == int:
                    default = float(default) if type(param[0]) == float else int(default)
                    value = st.slider(name, min_value=param[0], max_value=param[-1], step=param[1]-param[0], value=default)
                    selected_items[current_message["models"][st.session_state.hp_index]][name] = value
                else:
                    value = st.radio(name, options=param, index=param.index(default))
                    selected_items[current_message["models"][st.session_state.hp_index]][name] = default
            confirm_button = st.button("Confirm")
            if confirm_button and st.session_state.hp_index < len(current_message["models"]) - 1:
                st.session_state.history.append({"role": "assistant", "content": hp1[st.session_state.hp_index]})
                st.session_state.history.append({"role": "user", "content": f"Hyperparameter selections: {display_dict(selected_items[current_message['models'][st.session_state.hp_index]], hp=True)}"})
                st.session_state.selected_hps[current_message["models"][st.session_state.hp_index]] = selected_items[current_message["models"][st.session_state.hp_index]]
                log_event(logger,logging.INFO,"Hyperparameter step confirmed",hp_index=st.session_state.hp_index,)
                st.session_state.hp_index += 1
                st.rerun()
            elif confirm_button:
                st.session_state.history.append({"role": "assistant", "content": hp1[st.session_state.hp_index]})
                st.session_state.history.append({"role": "user", "content": f"Hyperparameter selections: {display_dict(selected_items[current_message['models'][st.session_state.hp_index]], hp=True)}"})
                st.session_state.selected_hps[current_message["models"][st.session_state.hp_index]] = selected_items[current_message["models"][st.session_state.hp_index]]
                for item in st.session_state.selected_hps:
                    for param in st.session_state.selected_hps[item]:
                        if current_message["hp_types"][item][param] == "float":
                            st.session_state.selected_hps[item][param] = float(st.session_state.selected_hps[item][param])
                        elif current_message["hp_types"][item][param] == "int":
                            st.session_state.selected_hps[item][param] = int(st.session_state.selected_hps[item][param])
                        elif current_message["hp_types"][item][param] == "bool":
                            st.session_state.selected_hps[item][param] = bool(st.session_state.selected_hps[item][param])
                        elif current_message["hp_types"][item][param] == "mix":
                            if current_message["hyperparameters"][item][param].index(st.session_state.selected_hps[item][param]) >= 2:
                                st.session_state.selected_hps[item][param] = float(st.session_state.selected_hps[item][param])
                log_event(logger, logging.INFO, "Hyperparameter selections submitted")
                st.session_state.hp_index = 0
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                response_payload = {response_tag: st.session_state.selected_hps}
                log_values_with_types(logger,logging.INFO,"hp continue_agent request",response_payload=response_payload,)
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        interrupt_payload = continue_agent(config, response_payload)
                log_values_with_types(logger,logging.INFO,"hp continue_agent interrupt",interrupt_payload=interrupt_payload,)
                if next_message["type"] == "multi-choice":
                    next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                    next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                elif next_message["type"] == "choice":
                    next_message["options"] = interrupt_payload["interrupt_message"]
                else:
                    next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                st.session_state.step += 1
                st.rerun()
        # elif message_type == "test":
        #     st.markdown("![Correlation Matrix](app/static/correlation_matrix.png)")

        elif message_type == "download":
            with st.chat_message("assistant"):
                st.write("Click here to download the model")
                with open("pickles/analyst_call_results.pkl", "rb") as file:
                    st.download_button("Download Model Pipeline", file, "model.pkl", "application/octet-stream")

        elif message_type == "model-choice":
            options = current_message["options"]
            log_values_with_types(logger,logging.INFO,"model-choice step current_message fields",options=options,)
            if "selected_option" not in st.session_state:
                st.session_state.selected_option = None
            if "report_content" not in st.session_state:
                st.session_state.report_content = None
            if "go_back" not in st.session_state:
                st.session_state.go_back = False
            cols = st.columns(len(options) + 1)
            for i in range(len(options)):
                with cols[i]:
                    if st.button(options[i]["slug"], key=f"btn_{i}"):
                        st.session_state.selected_option = options[i]["slug"]
                        st.session_state.report_content = None
            with cols[-1]:
                print("DEBUG: In Go Back column")
                log_event(logger, logging.INFO, "Entered Go Back column", extra={"step": st.session_state.get("step", "N/A"), "message_type": current_message.get("type", "N/A")})
                if st.button("Go Back"):
                    st.session_state.go_back = True

            if st.session_state.go_back:
                print("Go Back button pressed!")
                log_event(logger, logging.INFO, "Go Back button pressed")
                nodes = current_message['nodes']
                print(f"Nodes dictionary: {nodes}")
                log_event(logger, logging.INFO, "Nodes loaded", extra={"nodes_keys": list(nodes.keys())})
                for i, col in enumerate(nodes):
                    print(f"Rendering retrain button for {col} at index {i}")
                    if st.button(col, key=f"btn_retrain_{i}"):
                        print(f"[BTN_RETRAIN_{i}] {col} pressed. Setting selected_node and preparing to continue agent")
                        st.session_state.selected_node = current_message['nodes'][col]
                        print(f"st.session_state.selected_node set to: {st.session_state.selected_node}")
                        log_event(logger, logging.INFO, f"Retrain button {col} pressed", extra={"selected_node": st.session_state.selected_node})

                        response_payload = st.session_state.selected_node
                        print(f"Prepared response_payload: {response_payload}")
                        config = {"configurable": {"thread_id": st.session_state.session_id}}
                        print(f"Prepared config dict: {config}")
                        log_event(logger, logging.INFO, "About to call continue_agent", extra={"config": config, "response_payload": response_payload})
                        with st.chat_message("assistant"):
                            interrupt_payload = continue_agent(config, response_payload)
                        print(f"Received interrupt_payload: {interrupt_payload}")
                        log_event(logger, logging.INFO, "continue_agent returned", extra={"interrupt_payload": interrupt_payload})

                        found_step = False
                        for idx, msg in enumerate(st.session_state.messages_order):
                            print(f"Scanning message_list idx={idx}, response_tag={msg.get('response_tag')}")
                            if msg.get("response_tag") == st.session_state.selected_node:
                                st.session_state.step = idx
                                print(f"Set st.session_state.step to {idx} for response_tag {st.session_state.selected_node}")
                                found_step = True
                                break
                        log_event(logger, logging.INFO, "Step found in message_list", extra={"step": st.session_state.get("step"), "found_step": found_step})

                        new_message = st.session_state.messages_order[st.session_state.step]
                        print(f"New message at rerouted step: {new_message}")
                        log_event(logger, logging.INFO, "Loaded new_message", extra={"new_message_type": new_message.get("type"), "step": st.session_state.get("step")})

                        # Now propagate options/prompts based on message type
                        if new_message["type"] == "multi-choice" or new_message["type"] == "hp":
                            print(f"new_message is 'multi-choice' or 'hp'; next_message={new_message}")
                            if new_message["response_tag"] == "filter":
                                print("next_message.response_tag is 'filter'")
                                new_message["options"] = interrupt_payload["struct"][new_message["node_name"][0]]
                                new_message["prompts"] = interrupt_payload["struct"][new_message["node_name"][1]]
                                log_event(logger, logging.INFO, "Set options/prompts directly from struct for filter", extra={"options": new_message["options"], "prompts": new_message["prompts"]})
                            else:
                                print("next_message.response_tag is not 'filter' -- json loading output")
                                new_message["options"] = json.loads(interrupt_payload["struct"][new_message["node_name"][0]])
                                new_message["prompts"] = json.loads(interrupt_payload["struct"][new_message["node_name"][1]])
                                log_event(logger, logging.INFO, "Set options/prompts via json.loads", extra={"options": new_message["options"], "prompts": new_message["prompts"]})
                        elif new_message["type"] == "choice":
                            print("new_message type is CHOICE; assigning interrupt_message to options")
                            new_message["options"] = interrupt_payload["interrupt_message"]
                            log_event(logger, logging.INFO, "Set options to interrupt_message", extra={"options": new_message["options"]})
                        else:
                            print("new_message type is something else; loading options from struct")
                            new_message["options"] = json.loads(interrupt_payload["struct"][new_message["node_name"]])
                            log_event(logger, logging.INFO, "Set options via generic struct/json.loads", extra={"options": new_message["options"]})
                        print("Rerunning Streamlit (st.rerun)")
                        log_event(logger, logging.INFO, "Calling st.rerun() after retrain pathway")
                        st.session_state.go_back = False
                        st.rerun()
    
            if st.session_state.selected_option and st.session_state.report_content is None:
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                response_payload = {response_tag: st.session_state.selected_option}
                log_values_with_types(logger,logging.INFO,"model-choice continue_with_streaming request",response_payload=response_payload,)
                with st.spinner("Generating Report..."):
                    streamed_chunks = []
                    def capturing_stream():
                        for chunk in continue_with_streaming(config, response_payload):
                            streamed_chunks.append(chunk)
                            yield chunk
                    with st.chat_message("assistant"):
                        st.write_stream(capturing_stream())
                    st.session_state.report_content = "".join(streamed_chunks)
                    with open("/tmp/generated_report.md", "w", encoding="utf-8") as md_file:
                        md_file.write(st.session_state.report_content)

            if st.session_state.report_content:
                st.markdown("Click here to download the model report and file")
                st.download_button( "Download Model Report", st.session_state.report_content.encode("utf-8"), "report.md", "text/markdown")
                st.download_button("Download Model Pipeline", open("/tmp/pipeline.pkl", "rb"), "model.pkl", "application/octet-stream")
                st.markdown("""
                ```python
                import joblib
                pipeline = joblib.load(model.pkl)
                # load in your inference data
                ypred = pipeline.predict(inference_data)
                """)

        elif message_type == "popup":
            @st.dialog("Choose an API Provider", width = "large")
            def choose_model():
                cols = st.columns(6)
                st.markdown("It is recommended to use MistralAI `mistral-medium-3-5` model for this task, it is available free for inference. Other models might misbehave")
                buttons = ["OpenAI", "Anthropic", "Google Gemini", "Azure OpenAI", "Groq", "MistralAI"]
                for i, label in enumerate(buttons):
                    with cols[i]:
                        if st.button(label, use_container_width=True):
                            st.session_state.choice = label
                            st.rerun()
            @st.dialog("Set API Key")
            def set_key():
                if st.session_state.choice == "Azure OpenAI":
                    key = st.text_input("Enter your API key")
                    endpoint = st.text_input("Enter your Azure Endpoint e.g. https://YOUR-RESOURCE.openai.azure.com/")
                    deployment = st.text_input("Enter your Azure Deployment Name e.g. my-gpt4o-deployment")
                    api_version = st.text_input("Enter the API version e.g. 2024-02-01")
                else:
                    key = st.text_input("Enter your API key")
                    model_name = st.text_input("Enter your model name")

                if st.button("Confirm"):
                    st.session_state.key = key
                    if st.session_state.choice == "Azure OpenAI":
                        st.session_state.endpoint = endpoint
                        st.session_state.deployment = deployment
                        st.session_state.api_version = api_version
                    else:
                        st.session_state.model = model_name
                    st.session_state.step += 1
                    st.rerun()

            if "choice" not in st.session_state:
                st.session_state.choice = None
                choose_model()
            else:
                set_key()

        elif message_type == "cleaning":
            options = current_message["options"]['actions']
            actions = list(options.keys())
            prompt = current_message["options"]["prompts"][0]
            with st.chat_message("assistant"):
                st.markdown(prompt)
            selected = {action: [] for action in actions}
            for i, action in enumerate(actions):
                if options[action] != []:
                    st.markdown("**" + convert_from_snake_case(action) + "**")
                for col in options[action]: 
                    if st.checkbox(label=convert_from_snake_case(col), value=True, key=action + col):
                        selected[action].append(col)
            if st.button("Confirm"):
                st.session_state.history.append({"role": "assistant", "content": prompt})
                st.session_state.history.append({"role": "user", "content": "Cleaning selections: " + display_dict(selected)})
                log_event(logger,logging.INFO,"Cleaning selections submitted",selected_groups=len(selected),)
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                response_payload = {response_tag: selected}
                log_values_with_types(logger,logging.INFO,"cleaning continue_agent request",response_payload=response_payload,)
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        interrupt_payload = continue_agent(config, response_payload)
                log_values_with_types(logger,logging.INFO,"cleaning continue_agent interrupt",interrupt_payload=interrupt_payload,)
                
                if st.session_state.nulls <= 0:
                    st.session_state.step += 1
                next_message = st.session_state.messages_order[st.session_state.step + 1]
                if next_message["type"] == "multi-choice":
                    next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"][0]])
                    next_message["prompts"] = json.loads(interrupt_payload["struct"][next_message["node_name"][1]])
                elif next_message["type"] == "choice":
                    next_message["options"] = interrupt_payload["interrupt_message"]
                else:
                    next_message["options"] = json.loads(interrupt_payload["struct"][next_message["node_name"]])
                st.session_state.step += 1
                st.rerun()

