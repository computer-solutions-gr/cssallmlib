# %%
from cssallmlib.vectordb.chroma_db import ChromaManager

chroma_manager = ChromaManager()

# %%
chroma_manager.upsert_documents(
    documents=[
        ("Hello, world!", {"source": "test"}),
        ("My name is John Doe", {"source": "test"}),
    ],
)

# %%
chroma_manager.collection.get(
    ids=["fafc7e8a-aa83-4f8f-853c-79ac3e2b9b7c"],
    include=["embeddings", "documents", "metadatas"],
)

# %%
chroma_manager.collection.count()

# %%
chroma_manager.collection.get()

# %%
chroma_manager.client.list_collections()

# %%
