import os
import chromadb
from .operations import VectorDBManager
from loguru import logger
import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_PATH = "../chroma_db"
DEFAULT_COLLECTION_NAME = "default"


class ChromaManager(VectorDBManager):
    def __init__(
        self, path: str = DEFAULT_PATH, collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> None:
        """
        Initialize the ChromaManager with a specified path and collection name.

        :param path: The directory path where the Chroma database will persist.
        :param collection_name: The name of the collection to use or create.
        """
        super().__init__()

        self.client = chromadb.PersistentClient(path=path)
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
        logger.info(f"Chroma database initialized at {os.path.abspath(path)}")

    def _generate_ids(self, num_ids: int) -> list[str]:
        """
        Generate a list of unique IDs.

        :param num_ids: The number of IDs to generate.
        :return: A list of unique IDs.
        """
        return [str(uuid.uuid4()) for _ in range(num_ids)]

    def upsert_documents(self, documents: list[dict]) -> None:
        """
        Insert or update documents in the vector store.

        :param documents: A list of documents to be upserted.
        """
        try:
            _ids = self._generate_ids(len(documents))
            logger.info(_ids)

            inserted = self.vector_store.add_documents(
                documents=documents,
                ids=_ids,
            )
            logger.info(inserted)
            self.last_ids = inserted
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")

    def search_documents(
        self, query: str, k: int = 5, filter: dict = None, with_score: bool = False
    ) -> list:
        """
        Search for documents in the vector store based on a query.

        :param query: The search query string.
        :param k: The number of top results to return.
        :param filter: Optional filter criteria for the search.
        :param with_score: Whether to return the search results with similarity scores.
        :return: A list of search results, optionally with scores.
        """
        try:
            if with_score:
                return self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter
                )
            else:
                return self.vector_store.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_document(self, id: str) -> dict:
        """
        Retrieve a document from the vector store by its ID.

        :param id: The ID of the document to retrieve.
        :return: The document corresponding to the given ID.
        """
        try:
            return self.vector_store.get(ids=[id])
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None

    def count_documents(self) -> int:
        """
        Count the number of documents in the vector store.

        :return: The total number of documents.
        """
        try:
            return len(self.vector_store.get()["ids"])
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

    def get_all_documents(self) -> list[dict]:
        """
        Retrieve all documents from the vector store.

        :return: A list of all documents.
        """
        try:
            return self.vector_store.get()
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

    def list_collections(self) -> list[str]:
        """
        List all collections in the Chroma database.

        :return: A list of collection names.
        """
        try:
            return self.client.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
