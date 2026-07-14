import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config.env"))

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT","us-east-1") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index") 

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Paths (adjust as needed)
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR", "data")