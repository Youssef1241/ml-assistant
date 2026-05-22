from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Any, Dict, Union, Optional
from helpers import create_model_instance
import streamlit as st

load_dotenv()
# model = ChatGoogleGenerativeAI(
#     model=os.environ.get("GEMINI2_5"),
#     temperature=1.0, 
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# small_model = ChatGoogleGenerativeAI(
#     model=os.environ.get("GEMINI2_5"),
#     temperature=1.0, 
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )


# model =  ChatMistralAI(model="mistral-medium-3-5",)
# small_model = ChatMistralAI(model="mistral-medium-3-5",)
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

class FilterPayload(BaseModel):
    reasoning: str
    actions: Dict[str, Dict[str, str]]
    prompts: List[str]