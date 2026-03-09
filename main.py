
from pydantic_core.core_schema import general_after_validator_function
from analyst.nodes import analyst_call
from preprocessor.nodes import preprocessor_call
from state import should_continue, MessagesState, make_tool_node, ask_user_node, should_continue_with_user
from langgraph.graph import StateGraph, START, END
from analyst.tools import tools_by_name as analyst_tools
from preprocessor.tools import tools_by_name as preprocessor_tools
from langgraph.checkpoint.memory import InMemorySaver
import uuid



# Build workflow
agent_builder = StateGraph(MessagesState)
analyst_tool_node = make_tool_node(analyst_tools)
preprocessor_tool_node = make_tool_node(preprocessor_tools)

# Add nodes
agent_builder.add_node("analyst", analyst_call)
agent_builder.add_node("analyst_tools", analyst_tool_node)
agent_builder.add_node("preprocessor", preprocessor_call)
agent_builder.add_node("preprocessor_tools", preprocessor_tool_node)
agent_builder.add_node("user_input",ask_user_node)


# Add edges to connect nodes
agent_builder.add_edge(START, "analyst")
agent_builder.add_conditional_edges(
    "analyst",
    should_continue,
    ["analyst_tools", "preprocessor"]
)
agent_builder.add_edge("analyst_tools", "analyst")


agent_builder.add_conditional_edges(
    "preprocessor",
    should_continue_with_user,
    ["preprocessor_tools","user_input", END]
)
agent_builder.add_edge("preprocessor_tools", "preprocessor")
agent_builder.add_edge("user_input", "preprocessor")

# Compile the agent
checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}
from langchain.messages import HumanMessage, SystemMessage

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}

messages = [SystemMessage(content="start your task"), HumanMessage(content="Analyze the data")]

print("Starting agent stream...")
print("=" * 50)

# Process the stream - this will pause for user input when ask_user_node is called
for chunk in agent.stream({"messages": messages}, config):
    for node_name, node_output in chunk.items():
        print(f"\n[{node_name}]")
        if "messages" in node_output and len(node_output["messages"]) > 0:
            last_message = node_output["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print(f"Content: {last_message.content}")
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"Tool calls: {last_message.tool_calls}")
        else:
            print(f"Output: {node_output}")

print("\n" + "=" * 50)
print("Agent execution completed!")

