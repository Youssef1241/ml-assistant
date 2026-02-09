from langchain.agents.middleware.types import AgentState


from typing import Any


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from preprocessor.tools import tools
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=1.0, 
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

tools = [
      {"name": "replace_with_avg", "description": "Replace null values in a column with the average value for that column."},
      {"name": "drop_column", "description": "Drop a column from the dataset."},
      {"name": "drop_all_rows", "description": "Drop all rows that contain a null value in the specified column."},
]

model = create_agent(
    model = 'gemini-2.5-flash',
    temperature=1.0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
    middleware=[
        HumanInTheLoopMiddleware( 
            interrupt_on={tool["name"]: True for tool in tools}
        ),
    ],
)

model_with_tools = model.bind_tools(tools)