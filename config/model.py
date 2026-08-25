from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Any, Dict

load_dotenv()

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