import os
from dotenv import load_dotenv

# Set the base directory to the project's root, two levels up from this file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Đọc các biến môi trường từ Docker Compose
HOST_NAME = os.getenv("POSTGRES_HOST")
DB_NAME = os.getenv("POSTGRES_DB")
USER_NAME = os.getenv("POSTGRES_USER")
PWD = os.getenv("POSTGRES_PASSWORD")
PORT_ID = os.getenv("POSTGRES_PORT")

# Load API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print("DEBUG: All config variables loaded successfully.")
