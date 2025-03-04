import chromadb

from .operations import VectorDBManager

from loguru import logger

import uuid


class ChromaManager(VectorDBManager):

    def __init__(self, path: str = "../chroma_db", collection_name: str = "default"):

        super().__init__()

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert_documents(self, documents):
        _ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        logger.info(_ids)

        _documents = [document[0] for document in documents]
        logger.info(_documents)

        _metadatas = [document[1] for document in documents]
        logger.info(_metadatas)
        self.collection.add(
            ids=_ids,
            documents=_documents,
            metadatas=_metadatas,
        )
