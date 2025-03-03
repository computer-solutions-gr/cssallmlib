from loguru import logger
import pinecone
from .operations import VectorDBManager
import uuid


class PineconeManager(VectorDBManager):
    """Pinecone implementation of VectorDBManager"""

    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone connection

        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
            index_name (str): Name of the Pinecone index to use
        """
        super().__init__()
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

    def upsert_vectors(self, vectors, metadata=None):
        """
        Upsert vectors to Pinecone

        Args:
            vectors (list): List of (id, vector) tuples
            metadata (dict, optional): Dictionary mapping ids to metadata
        """
        if not vectors:
            logger.info("No vectors to upsert")
            return

        if not isinstance(vectors, list):
            raise ValueError("vectors should be a list of tuples")

        if not all(isinstance(t, tuple) and len(t) == 2 for t in vectors):
            raise ValueError("vectors should be a list of tuples of length 2")

        if metadata and not isinstance(metadata, dict):
            raise ValueError("metadata should be a dictionary")

        vector_list = []
        for id, vec in vectors:
            if not isinstance(id, str):
                raise ValueError("id should be a string")

            if not isinstance(vec, list):
                raise ValueError("vector should be a list")

            if (metadata) and (id in metadata):
                vector_list.append((id, vec, metadata[id]))
            else:
                vector_list.append((id, vec, {}))

        try:
            self.index.upsert(vectors=vector_list)
            logger.info(f"Successfully upserted {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def search_similar(self, query_vector, top_k=5):
        """
        Search for similar vectors in Pinecone

        Args:
            query_vector: Vector to search for
            top_k (int): Number of results to return

        Returns:
            list: List of (id, score) tuples for similar vectors
        """
        try:
            results = self.index.query(
                vector=query_vector, top_k=top_k, include_metadata=True
            )
            return [(match.id, match.score) for match in results.matches]
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise

    def embed_and_upsert(self, sentences, ids=None, metadata=None):
        """
        Create embeddings from sentences and upsert them

        Args:
            sentences (list): List of sentences to embed
            ids (list, optional): List of IDs for the sentences
            metadata (list, optional): List of metadata dictionaries for each sentence
        """
        try:
            if not isinstance(sentences, list):
                raise ValueError("sentences should be a list of strings")

            # Create embeddings
            embeddings = self.model.encode(sentences)

            # Check if embeddings are valid
            if not isinstance(embeddings, list) or not all(embeddings):
                raise ValueError("embeddings should be a list of vectors")

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in sentences]

            # Check if IDs are valid
            if not isinstance(ids, list) or not all(ids):
                raise ValueError("ids should be a list of strings")

            # Check if metadata is valid
            if metadata is not None:
                if not isinstance(metadata, list) or len(metadata) != len(sentences):
                    raise ValueError(
                        "metadata should be a list of dictionaries with the same length as sentences"
                    )
                for meta in metadata:
                    if not isinstance(meta, dict):
                        raise ValueError("each metadata item should be a dictionary")

            # Create vectors list
            vectors = list(zip(ids, embeddings))

            # Transform metadata list to dictionary indexed by IDs
            if metadata is not None:
                metadata_dict = {id_: meta for id_, meta in zip(ids, metadata)}
            else:
                metadata_dict = None

            # Upsert vectors to Pinecone
            self.upsert_vectors(vectors, metadata_dict)

            return ids
        except Exception as e:
            logger.error(f"Error embedding and upserting sentences: {str(e)}")
            raise
