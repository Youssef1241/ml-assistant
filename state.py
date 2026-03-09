from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.types import interrupt
from typing import Literal
from langgraph.graph import END
from langchain.messages import ToolMessage, HumanMessage



class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    struct: dict
    analysis: dict
    user_choice: dict
    update: dict

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
            observation = tool.invoke(tool_call["args"])

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
