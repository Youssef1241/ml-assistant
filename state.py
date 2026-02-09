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

def should_continue_with_user(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "preprocessor_tools"

    # Otherwise, we stop (reply to the user)
    return END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "analyst_tools"

    # Otherwise, we stop (reply to the user)
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
    question = last.tool_calls[0]["args"].get("question", "Provide input:")
    print("\nLLM: ",question)
    user_answer = interrupt(question)
    return {
        "messages": [HumanMessage(content=user_answer)]
    }