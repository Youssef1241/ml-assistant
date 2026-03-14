from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()



model = ChatGoogleGenerativeAI(
    model=os.environ.get("MODEL_NAME"),
    temperature=1.0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
