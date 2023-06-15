import os
import chromadb
from chromadb.config import Settings
from pathlib import Path

# def get_chroma_client(context:str):


# import openai

AGENT_DATA_PATH = "src/chatbot/agent_data"
AGENT_CHROMA_DATA_PATH = "src/chatbot/agent_data/chroma"
DATA_DIR = "src/chatbot/data"
CSV_DATA_DIR = "src/chatbot/data/CSV"

Path(DATA_DIR).mkdir(exist_ok=True)
Path(CSV_DATA_DIR).mkdir(exist_ok=True)

Path(AGENT_DATA_PATH).mkdir(exist_ok=True)
Path(AGENT_CHROMA_DATA_PATH).mkdir(exist_ok=True)


PARENT_SRC = os.getcwd()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CLIENT = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=AGENT_CHROMA_DATA_PATH)
)
AGENT_SETTINGS = {
    "CoversationHistoryLen": 3,
    "HumanName": "SHUVRA NEEL ROY",
    "AgentName": "Luci",
}


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
