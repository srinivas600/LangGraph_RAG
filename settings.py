
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    OPENAI_MODEL = "gpt-4o"

    # Hugging Face Configuration
    EMBEDDING_MODEL = "C:/projects/RAG_LangGraph/new/E5_large"

    # Chroma DB Configuration
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "knowledge_base"

    # Database Configuration
    DATABASE_URL = "sqlite:///C:/projects/RAG_LangGraph/new/chatbot.db"

    # Chat Configuration
    MAX_CHAT_HISTORY = 5

    # FastAPI Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # Streamlit Configuration
    STREAMLIT_PORT = 8501

    # DeepEval Configuration
    EVALUATION_MODEL = "gpt-4o"
