import chromadb
from .operations import VectorDBManager
from loguru import logger
import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaManager(VectorDBManager):
    def __init__(self, path: str = "../chroma_db", collection_name: str = "default"):
        super().__init__()

        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory=path,
        )

    def upsert_documents(self, documents):
        _ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        logger.info(_ids)

        inserted = self.vector_store.add_documents(
            documents=documents,
            ids=_ids,
        )
        logger.info(inserted)
        self.last_ids = inserted


    def search_documents(self, query: str, k: int = 5, filter: dict = None, with_score: bool = False):
        if with_score:
            return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
        else:
            return self.vector_store.similarity_search(query, k=k, filter=filter)

    def get_document(self, id: str):
        return self.vector_store.get(ids=[id])

    def count_documents(self):
        return len(self.vector_store.get()['ids'])

    def get_all_documents(self):
        return self.vector_store.get()

    def list_collections(self):
        return self.client.list_collections()
