# %%
from cssallmlib.vectordb.chroma_db import ChromaManager

chroma_manager = ChromaManager()

# %%
# chroma_manager.upsert_documents(
#     documents=[
#         ("Hello, world!", {"source": "test"}),
#         ("My name is John Doe", {"source": "test"}),
#     ],
# )
chroma_manager.embed_and_upsert(
    documents=[
        "Hello, world!",
        "My name is John Doe",
    ],
    metadata=[{"source": "test"}, {"source": "test"}],
)

# %%
chroma_manager.collection.get(
    ids=["84603a7e-d4e1-4800-bf27-a7a8f26e9e84"],
    include=["embeddings", "documents", "metadatas"],
)

# %%
chroma_manager.collection.count()

# %%
chroma_manager.collection.get()

# %%
chroma_manager.client.list_collections()

# %%
