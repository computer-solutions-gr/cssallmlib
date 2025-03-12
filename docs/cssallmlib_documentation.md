# CSSALLMLIB Module Documentation

## Overview
The `cssallmlib` module provides utilities for vector database operations and core functionalities for LLM (Large Language Model) operations. It includes classes for managing vector databases and processing prompts for LLMs.

## Modules

### 1. `core.py`
This module contains core functionalities for LLM operations.

#### Classes

- **`LLMHelper`**
  - **Description**: Provides helper functions for processing prompts for LLM consumption.
  
  - **Methods**:
    - `__init__()`
      - **Description**: Initializes the LLM Helper.
    - `process_prompt(prompt: str) -> str`
      - **Description**: Processes and prepares a prompt for LLM consumption.
      - **Parameters**:
        - `prompt (str)`: The input prompt to process.
      - **Returns**: Processed prompt ready for LLM input.

### 2. `vectordb` Package
This package provides utilities for vector database manipulation.

#### Modules

- **`operations.py`**
  - **Classes**
    - **`VectorDBManager`**
      - **Description**: Base class for vector database operations.
      
      - **Methods**:
        - `_generate_ids(num_ids: int) -> list[str]`
          - **Description**: Generates a list of unique IDs.
          - **Parameters**:
            - `num_ids (int)`: The number of IDs to generate.
          - **Returns**: A list of unique IDs.
        - `upsert_documents(documents: list[dict]) -> None`
          - **Description**: Abstract method to insert or update documents in the vector store.
          - **Parameters**:
            - `documents (list[dict])`: A list of documents to be upserted.
        - `search_documents(query: str, k: int = 5, filter: dict = None, with_score: bool = False) -> list`
          - **Description**: Abstract method to search for documents in the vector store based on a query.
          - **Parameters**:
            - `query (str)`: The search query string.
            - `k (int)`: The number of top results to return.
            - `filter (dict)`: Optional filter criteria for the search.
            - `with_score (bool)`: Whether to return the search results with similarity scores.
          - **Returns**: A list of search results, optionally with scores.
        - `embed_and_upsert(sentences, ids=None, metadata=None)`
          - **Description**: Creates embeddings from sentences and upserts them.
          - **Parameters**:
            - `sentences (list)`: List of sentences to embed.
            - `ids (list, optional)`: List of IDs for the sentences.
            - `metadata (dict, optional)`: Dictionary mapping ids to metadata.
        - `get_document(id: str) -> dict`
          - **Description**: Abstract method to retrieve a document from the vector store by its ID.
          - **Parameters**:
            - `id (str)`: The ID of the document to retrieve.
          - **Returns**: The document corresponding to the given ID.
        - `count_documents() -> int`
          - **Description**: Abstract method to count the number of documents in the vector store.
          - **Returns**: The total number of documents.
        - `get_all_documents() -> list[dict]`
          - **Description**: Abstract method to retrieve all documents from the vector store.
          - **Returns**: A list of all documents.
        - `list_collections() -> list[str]`
          - **Description**: Abstract method to list all collections in the vector database.
          - **Returns**: A list of collection names.

- **`chroma_db.py`**
  - **Classes**
    - **`ChromaManager`**
      - **Description**: Manages Chroma vector database operations.
      
      - **Methods**:
        - `__init__(path: str = DEFAULT_PATH, collection_name: str = DEFAULT_COLLECTION_NAME) -> None`
          - **Description**: Initializes the ChromaManager with a specified path and collection name.
          - **Parameters**:
            - `path (str)`: The directory path where the Chroma database will persist.
            - `collection_name (str)`: The name of the collection to use or create.
        - `_generate_ids(num_ids: int) -> list[str]`
          - **Description**: Generates a list of unique IDs.
          - **Parameters**:
            - `num_ids (int)`: The number of IDs to generate.
          - **Returns**: A list of unique IDs.
        - `upsert_documents(documents: list[dict]) -> None`
          - **Description**: Inserts or updates documents in the vector store.
          - **Parameters**:
            - `documents (list[dict])`: A list of documents to be upserted.
        - `search_documents(query: str, k: int = 5, filter: dict = None, with_score: bool = False) -> list`
          - **Description**: Searches for documents in the vector store based on a query.
          - **Parameters**:
            - `query (str)`: The search query string.
            - `k (int)`: The number of top results to return.
            - `filter (dict)`: Optional filter criteria for the search.
            - `with_score (bool)`: Whether to return the search results with similarity scores.
          - **Returns**: A list of search results, optionally with scores.
        - `get_document(id: str) -> dict`
          - **Description**: Retrieves a document from the vector store by its ID.
          - **Parameters**:
            - `id (str)`: The ID of the document to retrieve.
          - **Returns**: The document corresponding to the given ID.
        - `count_documents() -> int`
          - **Description**: Counts the number of documents in the vector store.
          - **Returns**: The total number of documents.
        - `get_all_documents() -> list[dict]`
          - **Description**: Retrieves all documents from the vector store.
          - **Returns**: A list of all documents.
        - `list_collections() -> list[str]`
          - **Description**: Lists all collections in the Chroma database.
          - **Returns**: A list of collection names. 