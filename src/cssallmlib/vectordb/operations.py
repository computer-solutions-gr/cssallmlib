from loguru import logger
import uuid


class VectorDBManager:
    """Base class for vector database operations"""

    def __init__(self):
        logger.info("Initializing VectorDB Manager")
        self.embedding_model = "all-MiniLM-L6-v2"

    def _generate_ids(self, num_ids: int) -> list[str]:
        """
        Generate a list of unique IDs.

        :param num_ids: The number of IDs to generate.
        :return: A list of unique IDs.
        """
        return [str(uuid.uuid4()) for _ in range(num_ids)]

    def upsert_documents(self, documents: list[dict]) -> None:
        """
        Abstract method to insert or update documents in the vector store.

        :param documents: A list of documents to be upserted.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def search_documents(
        self, query: str, k: int = 5, filter: dict = None, with_score: bool = False
    ) -> list:
        """
        Abstract method to search for documents in the vector store based on a query.

        :param query: The search query string.
        :param k: The number of top results to return.
        :param filter: Optional filter criteria for the search.
        :param with_score: Whether to return the search results with similarity scores.
        :return: A list of search results, optionally with scores.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def embed_and_upsert(self, sentences, ids=None, metadata=None):
        """
        Create embeddings from sentences and upsert them

        Args:
            sentences (list): List of sentences to embed
            ids (list, optional): List of IDs for the sentences
            metadata (dict, optional): Dictionary mapping ids to metadata
        """
        pass

    def get_document(self, id: str) -> dict:
        """
        Abstract method to retrieve a document from the vector store by its ID.

        :param id: The ID of the document to retrieve.
        :return: The document corresponding to the given ID.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def count_documents(self) -> int:
        """
        Abstract method to count the number of documents in the vector store.

        :return: The total number of documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_all_documents(self) -> list[dict]:
        """
        Abstract method to retrieve all documents from the vector store.

        :return: A list of all documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def list_collections(self) -> list[str]:
        """
        Abstract method to list all collections in the vector database.

        :return: A list of collection names.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
