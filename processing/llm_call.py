from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph.state import RunnableConfig
from langgraph.types import interrupt
from helpers import create_model_instance
import time
import logging
from logging_utils import get_logger, log_event
logger = get_logger(__name__)

def reporter_call(state: dict, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    log_event(logger, logging.INFO, "Reporter call start")
    context = "Context: \n"
    context += str(state['update']['model_metrics']) + "\n\n"
    messages_to_send = [
        SystemMessage(content=context),
        HumanMessage(content=
        """
                    You are an expert data scientist and handler tasked with recommending actions to a user
                    Your instructions: 
                        - The context provides model specs and their evaluation metrics. Recommend the most promising model for the user to choose

                        - The model specs will be in the slug format: scaling_method-model-imbalance_handling_method, in this order

                        - You will create a small comparison report for the user for the three models, and their performances
                        you have a bar chart comparing the models and their metrics in this address: app/static/model_metrics_barplot.png, you have a spider diagram comparing the models in this address: app/static/spider_comparison.png (only show the spider diagram if there are more than three metrics chosen, else just ignore it), and a roc-auc curve comparing the models in this address: app/static/roc_curve.png, you can use their addresses in their markdown

                        at the end of the small report, you will give your recommendation for the user to choose
                        - dont put the raw confusion matrix in the report, you can refer to it or mention numbers in it but dont put it in raw

                        - Use markdown for good stylization and professional language

                        """)
    ] 
    started_at = time.perf_counter()
    model = create_model_instance(state["model_info"])
    result = None
    streamed = False
    for chunk in model.stream(messages_to_send, config):
        streamed = True
        result = chunk if result is None else result + chunk
    if not streamed:
        # Safety fallback so downstream logic never receives None.
        result = model.invoke(messages_to_send)

    log_event(logger,logging.INFO,"Reporter call complete",elapsed_ms=int((time.perf_counter() - started_at) * 1000),)
    
    return {
        "messages": [result],
    }

def ask_user(state: dict):
    """ Ask the user for their choice """
    log_event(logger, logging.INFO, "Asking user to choose model")
    user_choice = interrupt({"struct": state['update']['model_metrics']})
    state_dict = state["user_choice"]
    if type(user_choice) == str:
        state_dict["retrain"] = user_choice
        return {"user_choice": {"retrain": user_choice},}
    else:
        state_dict["chooseone"] = user_choice
        return {"user_choice": {"chooseone": user_choice},}

def retrain_interrupt(state: dict):
    """ Ask the user for their choice """
    log_event(logger, logging.INFO, "rerouting retrain interrupt")
    user_choice = interrupt({"struct": "test"})
    return {
        "user_choice": {"retrain": user_choice},
    }

def report_generator(state: dict, config: RunnableConfig):
    """ Generate the report """
    log_event(logger, logging.INFO, "Report generation call start")
    context = "Context: \n"
    context += "Analysis report: "+ str(state['messages'][2].content)  + "\n\n"
    context += "User choices: "+ str(state['user_choice']) + "\n\n"
    context += "Evaluation report: "+ str(state["messages"][-1].content) + "\n\n"
    context += "Model Chosen: "+ str(state["user_choice"]["chooseone"]) + "\n\n"
    messages_to_send = [
        SystemMessage(content=context),
        HumanMessage(content=
        """
                    You are an expert data scientist and handler tasked generating machine learning reports
                    Your task is to generate a report for the user, based on the analysis, and evaluation reports given as as well as the user choices
                    You are to state any processes that have been done to the data, explain the processes done to the data, and any processes done before the model training
                    - dont put the raw confusion matrix in the report, you can refer to it or mention numbers in it but dont put it in raw
                    - do not add an appendix in the end, just normal conclusion
                    - when explaining why the final model was chosen, mention the pros and cons of that model, not why it was chosen, since the user chooses it not you
                        """
        )
        # SystemMessage(
        #     content=context + """
        #             You are an expert data scientist and handler tasked generating machine learning reports
        #             Your task is to generate a report for the user
        #             MAKE IT A VERY SHORT REPORT SINCE THIS IS A TESTING PHASE TO AVOID COST

        #                 """
        # )

    ] 
    started_at = time.perf_counter()
    model = create_model_instance(state["model_info"])
    result = None
    streamed = False
    for chunk in model.stream(messages_to_send, config):
        streamed = True
        result = chunk if result is None else result + chunk
    if not streamed:
        # Safety fallback so downstream logic never receives None.
        result = model.invoke(messages_to_send)
    
    # result = model.invoke(messages_to_send)
    # result = use_persistent(model,"report_generator_results.pkl", messages_to_send)
    log_event(logger,logging.INFO,"Report generation call complete",elapsed_ms=int((time.perf_counter() - started_at) * 1000),)
    
    return {
        "messages": [result],
    }