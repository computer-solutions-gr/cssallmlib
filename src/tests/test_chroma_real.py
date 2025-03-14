import time
from loguru import logger
import pytest
from cssallmlib.vectordb.chroma_db import ChromaManager
from langchain_core.documents import Document
import tempfile
import shutil


def import_documents(chroma_manager: ChromaManager):
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
        id=1,
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
        id=2,
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
        id=3,
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
        id=4,
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
        id=5,
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
        id=6,
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
        id=7,
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
        id=8,
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
        id=9,
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
        id=10,
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    chroma_manager.upsert_documents(documents)


@pytest.fixture(scope="module")
def chroma_manager(request):
    temp_dir = tempfile.mkdtemp()
    manager = ChromaManager(collection_name="test_collection", path=temp_dir)
    import_documents(manager)

    def teardown():
        # First, reset the collection to ensure data is saved
        try:
            manager.vector_store.reset_collection()
        except Exception as e:
            logger.warning(f"Failed to reset collection: {e}")
        
        # Try multiple approaches to close the ChromaDB client
        try:
            # Try to clean up ChromaDB resources using different approaches
            if hasattr(manager, 'client') and manager.client:
                # Approach 1: Try to stop the system
                try:
                    if hasattr(manager.client, '_system'):
                        manager.client._system.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop ChromaDB system: {e}")
                
                # Approach 2: Try to close the client
                try:
                    if hasattr(manager.client, 'close'):
                        manager.client.close()
                except Exception as e:
                    logger.warning(f"Failed to close ChromaDB client: {e}")
                
                # Approach 3: Try to clear system cache
                try:
                    import chromadb
                    if hasattr(chromadb, 'clear_system_cache'):
                        chromadb.clear_system_cache()
                except Exception as e:
                    logger.warning(f"Failed to clear ChromaDB system cache: {e}")
            
            # Explicitly set references to None to help garbage collection
            if hasattr(manager, 'vector_store'):
                manager.vector_store = None
            if hasattr(manager, 'client'):
                manager.client = None
            if hasattr(manager, 'collection'):
                manager.collection = None
        except Exception as e:
            logger.warning(f"Error during ChromaDB cleanup: {e}")
        
        # Wait longer before attempting to delete
        time.sleep(2)
        
        # Try to remove the directory with retries and longer timeouts
        for attempt in range(5):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Successfully removed temp directory: {temp_dir}")
                break
            except Exception as e:
                logger.warning(f"Failed to remove temp directory (attempt {attempt+1}/5): {e}")
                # Increase wait time with each attempt
                time.sleep(2 * (attempt + 1))
        else:
            logger.error(f"Could not remove temp directory after 5 attempts: {temp_dir}")
            # On Windows, sometimes we need to defer cleanup to a later time
            try:
                import atexit
                atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
                logger.info(f"Registered temp directory {temp_dir} for cleanup at exit")
            except Exception as e:
                logger.error(f"Failed to register directory for cleanup at exit: {e}")
    
    request.addfinalizer(teardown)
    
    return manager


def test_get_and_count_documents(chroma_manager):
    assert chroma_manager.count_documents() == 10
    assert chroma_manager.list_collections() is not None


def test_get_all_documents(chroma_manager):
    assert chroma_manager.get_all_documents() is not None


def test_search_for_pancakes(chroma_manager):
    assert chroma_manager.search_documents("pancakes") is not None
    assert chroma_manager.search_documents("pancakes", with_score=True) is not None



def test_search_langchain_documents(chroma_manager):
    search = chroma_manager.search_documents(
        "LangChain provides abstractions to make working with LLMs easy", k=2
    )
    assert len(search) == 2
    assert "Building an exciting new project with LangChain" in search[0].page_content
    assert (
        "LangGraph is the best framework for building stateful, agentic applications!"
        in search[1].page_content
    )

def test_search_weather_documents(chroma_manager):
    search = chroma_manager.search_documents(
        "Will it be hot tomorrow?",
        k=2,
        filter={"source": "news"},
        with_score=True,
    )
    assert len(search) == 2
    assert (
        "The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees."
        in search[0][0].page_content
    )
    assert (
        "The stock market is down 500 points today due to fears of a recession."
        in search[1][0].page_content
    )
    assert search[0][1] == pytest.approx(0.916557)
