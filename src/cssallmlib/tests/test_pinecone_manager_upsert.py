import pytest
from unittest.mock import patch
from cssallmlib.vectordb.pinecone_db import PineconeManager

@pytest.fixture
def mock_pinecone():
    with patch('pinecone.init') as mock_init, patch('pinecone.Index') as mock_index:
        mock_instance = mock_index.return_value
        yield {"init": mock_init, "index": mock_instance}

@pytest.fixture
def pinecone_manager(mock_pinecone):
    return PineconeManager(
        api_key="test-key",
        environment="test-env",
        index_name="test-index"
    )

def test_upsert_vectors_with_metadata(pinecone_manager, mock_pinecone):
    vectors = [
        ("id1", [0.1, 0.2, 0.3]),
        ("id2", [0.4, 0.5, 0.6])
    ]
    metadata = {
        "id1": {"type": "doc1"},
        "id2": {"type": "doc2"}
    }
    
    pinecone_manager.upsert_vectors(vectors, metadata)
    
    expected_vector_list = [
        ("id1", [0.1, 0.2, 0.3], {"type": "doc1"}),
        ("id2", [0.4, 0.5, 0.6], {"type": "doc2"})
    ]
    mock_pinecone["index"].upsert.assert_called_once_with(vectors=expected_vector_list)

def test_upsert_vectors_without_metadata(pinecone_manager, mock_pinecone):
    vectors = [
        ("id1", [0.1, 0.2, 0.3]),
        ("id2", [0.4, 0.5, 0.6])
    ]
    
    pinecone_manager.upsert_vectors(vectors)
    
    expected_vector_list = [
        ("id1", [0.1, 0.2, 0.3], {}),
        ("id2", [0.4, 0.5, 0.6], {})
    ]
    mock_pinecone["index"].upsert.assert_called_once_with(vectors=expected_vector_list)

def test_upsert_vectors_partial_metadata(pinecone_manager, mock_pinecone):
    vectors = [
        ("id1", [0.1, 0.2, 0.3]),
        ("id2", [0.4, 0.5, 0.6])
    ]
    metadata = {
        "id1": {"type": "doc1"}
    }
    
    pinecone_manager.upsert_vectors(vectors, metadata)
    
    expected_vector_list = [
        ("id1", [0.1, 0.2, 0.3], {"type": "doc1"}),
        ("id2", [0.4, 0.5, 0.6], {})
    ]
    mock_pinecone["index"].upsert.assert_called_once_with(vectors=expected_vector_list)

def test_upsert_vectors_error(pinecone_manager, mock_pinecone):
    vectors = [("id1", [0.1, 0.2, 0.3])]
    mock_pinecone["index"].upsert.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        pinecone_manager.upsert_vectors(vectors)
    mock_pinecone["index"].upsert.assert_called_once()