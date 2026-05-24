def use_persistent(model, path, messages):
    import os
    import pickle
    pkls_folder = "pickles"
    os.makedirs(pkls_folder, exist_ok=True)
    pkls_path = os.path.join(pkls_folder, path)
    if os.path.exists(pkls_path):
        with open(pkls_path, "rb") as f:
            result = pickle.load(f)
    # else:
        # return "No results found"
    else:
        result = model.invoke(messages)
        with open(pkls_path, "wb") as f:
            pickle.dump(result, f)
    return result

def create_model_instance(model_info):
    import streamlit as st
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mistralai import ChatMistralAI
    choice = model_info["choice"]
    if choice == "Azure OpenAI":
        llm_model = AzureChatOpenAI(
            api_version=model_info["api_version"],
            azure_deployment=model_info["deployment"],
            endpoint=model_info["endpoint"],
            api_key=model_info["api_key"],
        )
    elif choice == "OpenAI":
        llm_model = ChatOpenAI(api_key=model_info["api_key"], model=model_info["model"],)
    elif choice == "Anthropic":
        llm_model = ChatAnthropic(api_key=model_info["api_key"], model=model_info["model"])
    elif choice == "Groq":
        llm_model = ChatGroq(api_key=model_info["api_key"], model=model_info["model"])
    elif choice == "Google Gemini":
        llm_model = ChatGoogleGenerativeAI(api_key=model_info["api_key"], model=model_info["model"])
    elif choice == "MistralAI":
        llm_model = ChatMistralAI(api_key=model_info["api_key"], model=model_info["model"], max_retries = 5)
    return llm_model
