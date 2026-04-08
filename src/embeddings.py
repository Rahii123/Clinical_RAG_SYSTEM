import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def get_embedding_function():
    """
    Returns the high-reliability OpenAI embedding function (1536 dims).
    OpenAI is used for production bulk indexing to avoid Gemini rate limits.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if openai_key:
        print("🔹 Initializing OpenAIEmbeddings (1536 dims) - BEST FOR BULK")
        return OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=openai_key
        )
    elif gemini_key:
        print("🔹 Initializing GoogleGenerativeAIEmbeddings (3072 dims) - FALLBACK")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=gemini_key
        )
    else:
        raise ValueError("❌ No API Keys found! Place OPENAI_API_KEY in your .env file.")
