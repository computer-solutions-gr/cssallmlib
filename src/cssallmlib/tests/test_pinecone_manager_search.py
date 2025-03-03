import pytest
from unittest.mock import Mock, patch
from cssallmlib.vectordb.pinecone_db import PineconeManager

@pytest.fixture
def mock_pinecone():
    with patch('pinecone.Pinecone') as mock_init, patch('pinecone.Index') as mock_index:
        mock_instance = mock_index.return_value
        yield {"init": mock_init, "index": mock_instance}

@pytest.fixture
def pinecone_manager(mock_pinecone):
    return PineconeManager(
        api_key="test-key",
        environment="test-env",
        index_name="test-index"
    )

def test_search_similar_success(pinecone_manager, mock_pinecone):
    query_vector = [0.1, 0.2, 0.3]
    mock_response = Mock()
    mock_response.matches = [
        Mock(id="id1", score=0.9),
        Mock(id="id2", score=0.8),
        Mock(id="id3", score=0.7)
    ]
    mock_pinecone["index"].query.return_value = mock_response

    results = pinecone_manager.search_similar(query_vector, top_k=3)

    assert len(results) == 3
    assert results[0] == ("id1", 0.9)
    assert results[1] == ("id2", 0.8)
    assert results[2] == ("id3", 0.7)
    mock_pinecone["index"].query.assert_called_once_with(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

def test_search_similar_default_top_k(pinecone_manager, mock_pinecone):
    query_vector = [0.1, 0.2, 0.3]
    mock_response = Mock()
    mock_response.matches = [Mock(id=f"id{i}", score=0.9-i*0.1) for i in range(5)]
    mock_pinecone["index"].query.return_value = mock_response

    results = pinecone_manager.search_similar(query_vector)

    assert len(results) == 5
    mock_pinecone["index"].query.assert_called_once_with(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

def test_search_similar_empty_results(pinecone_manager, mock_pinecone):
    query_vector = [0.1, 0.2, 0.3]
    mock_response = Mock()
    mock_response.matches = []
    mock_pinecone["index"].query.return_value = mock_response

    results = pinecone_manager.search_similar(query_vector)

    assert len(results) == 0
    mock_pinecone["index"].query.assert_called_once()

def test_search_similar_error(pinecone_manager, mock_pinecone):
    query_vector = [0.1, 0.2, 0.3]
    mock_pinecone["index"].query.side_effect = Exception("Test error")

    with pytest.raises(Exception):
        pinecone_manager.search_similar(query_vector)
    mock_pinecone["index"].query.assert_called_once()