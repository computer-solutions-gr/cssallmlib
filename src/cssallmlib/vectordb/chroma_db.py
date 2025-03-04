import chromadb

from .operations import VectorDBManager

from loguru import logger

import uuid


class ChromaManager(VectorDBManager):

    def __init__(self, path: str = "../chroma_db", collection_name: str = "default"):

        super().__init__()

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def embed_and_upsert(self, documents, ids=None, metadata=None):
        """
        Create embeddings from documents and upsert them

        Args:
            documents (list): List of documents to embed
            ids (list, optional): List of IDs for the documents
            metadata (list, optional): List of metadata dictionaries for each document
        """
        try:
            if not isinstance(documents, list):
                raise ValueError("documents should be a list of strings")

            # Create embeddings
            embeddings = self.model.encode(documents)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]

            # Check if IDs are valid
            if not isinstance(ids, list) or not all(ids):
                raise ValueError("ids should be a list of strings")

            # Check if metadata is valid
            if metadata is not None:
                if not isinstance(metadata, list) or len(metadata) != len(documents):
                    raise ValueError(
                        "metadata should be a list of dictionaries with the same length as documents"
                    )
                for meta in metadata:
                    if not isinstance(meta, dict):
                        raise ValueError("each metadata item should be a dictionary")

            # Add documents to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadata if metadata else [{}] * len(documents)
            )

            return ids

        except Exception as e:
            logger.error(f"Error embedding and upserting documents: {str(e)}")
            raise
