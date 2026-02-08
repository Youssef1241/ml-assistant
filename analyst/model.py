from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
