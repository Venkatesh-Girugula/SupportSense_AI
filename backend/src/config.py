import os
from pathlib import Path
from dotenv import load_dotenv

# Get the backend root directory (backend/)
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the .env file in the backend root
env_path = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    """Application configuration container."""
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Model Configurations
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Directory and DB Paths (resolved to absolute paths)
    DOCS_DIR = BACKEND_DIR / os.getenv("DOCS_DIR", "docs")
    DB_PATH = BACKEND_DIR / os.getenv("DB_PATH", "faiss_index")
    
    # Logic Thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))
    
    # Log Severity Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def print_config(cls) -> str:
        """Helper to print safe representation of config settings for UI/logs."""
        return (
            f"LLM_MODEL: {cls.LLM_MODEL}\n"
            f"EMBEDDING_MODEL: {cls.EMBEDDING_MODEL}\n"
            f"DOCS_DIR: {cls.DOCS_DIR}\n"
            f"DB_PATH: {cls.DB_PATH}\n"
            f"CONFIDENCE_THRESHOLD: {cls.CONFIDENCE_THRESHOLD}\n"
            f"GEMINI_API_KEY: {'[SET]' if cls.GEMINI_API_KEY else '[MISSING]'}"
        )
# Ensure directories exist
Config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
