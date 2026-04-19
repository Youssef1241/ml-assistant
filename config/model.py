from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Any, Dict, Union, Optional

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

from typing import List, Dict
from pydantic import BaseModel


class Options(BaseModel):
    reasoning: str
    actions: Dict[str, List[str]]

class HPOptions(BaseModel):
    reasoning: str
    actions: Dict[str, Dict[str, Any]]
    
class Prompts(BaseModel):
    prompts: List[str]

class FullPayload(BaseModel):
    reasoning: str
    actions: Dict[str, List[str]]
    prompts: List[str]