from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Dict, Union, Optional

load_dotenv()
model = ChatGoogleGenerativeAI(
    model=os.environ.get("MODEL_NAME"),
    temperature=1.0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

small_model = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI2.5LITE"),
    temperature=1.0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from typing import Dict, Union
from pydantic import BaseModel, RootModel

class ValueType(RootModel[Dict[str, int]]):
    pass

class Options(BaseModel):
    recommendations: Dict[str, Union[int, ValueType]]


struct_model = small_model.with_structured_output(Options)