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
        try:
            vector_list = [
                (id, vec, metadata.get(id, {})) for id, vec in vectors
            ] if metadata else [(id, vec, {}) for id, vec in vectors]
            
            # Ensure metadata contains all ids
            if metadata:
                for id, vec in vectors:
                    if id not in metadata:
                        metadata[id] = {}
            
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
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
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
            metadata (dict, optional): Dictionary mapping ids to metadata
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
            
            # Create vectors list
            vectors = list(zip(ids, embeddings))
            
            # Check if vectors are valid
            if not isinstance(vectors, list) or not all(vectors):
                raise ValueError("vectors should be a list of tuples")
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {id_: {'text': sent} for id_, sent in zip(ids, sentences)}
            else:
                # Check if metadata is valid
                if not isinstance(metadata, dict) or not all(metadata):
                    raise ValueError("metadata should be a dictionary")
                
                # Ensure metadata contains all ids
                for id_ in ids:
                    if id_ not in metadata:
                        metadata[id_] = {}  # Initialize with empty dict if not present
                # Update metadata to use generated IDs as keys
                metadata = {id_: metadata.get(id_, {}) for id_ in ids}
            
            # Upsert vectors
            self.upsert_vectors(vectors=vectors, metadata=metadata)
            
            logger.info(f"Successfully embedded and upserted {len(sentences)} sentences")
            return ids
            
        except Exception as e:
            logger.error(f"Error in embed_and_upsert: {str(e)}")
            raise
