from dotenv import load_dotenv
from cssallmlib.vectordb.pinecone_db import PineconeManager
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv("PINECONE_API_KEY", "default_api_key")
environment = os.getenv("PINECONE_ENVIRONMENT", "default_environment")
index_name = os.getenv("PINECONE_INDEX_NAME", "default_index_name")

print(f"API Key: {api_key if api_key else 'Not found in .env'}")
print(f"Environment: {environment if environment else 'Not found in .env'}")
print(f"Index Name: {index_name if index_name else 'Not found in .env'}")

# Instantiate PineconeManager
pinecone_manager = PineconeManager(
    api_key=api_key, environment=environment, index_name=index_name
)

sentences = ["Hello world", "Test sentence"]

ids = pinecone_manager.embed_and_upsert(sentences)

print(ids)
