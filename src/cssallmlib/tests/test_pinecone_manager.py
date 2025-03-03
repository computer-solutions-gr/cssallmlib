import pytest
from unittest.mock import MagicMock, patch
from cssallmlib.vectordb.pinecone_db import PineconeManager

@pytest.fixture
def pinecone_manager():
    with patch('pinecone.init'), patch('pinecone.Index') as MockIndex:
        mock_index = MockIndex.return_value
        manager = PineconeManager(api_key="fake_api_key", environment="fake_env", index_name="fake_index")
        return manager, mock_index

def test_upsert_vectors(pinecone_manager):
    manager, mock_index = pinecone_manager
    vectors = [("id1", [0.1, 0.2, 0.3]), ("id2", [0.4, 0.5, 0.6])]
    metadata = {"id1": {"meta": "data1"}, "id2": {"meta": "data2"}}
    
    manager.upsert_vectors(vectors, metadata)
    
    mock_index.upsert.assert_called_once()
    assert mock_index.upsert.call_args[1]['vectors'] == [
        ("id1", [0.1, 0.2, 0.3], {"meta": "data1"}),
        ("id2", [0.4, 0.5, 0.6], {"meta": "data2"})
    ]

def test_search_similar(pinecone_manager):
    manager, mock_index = pinecone_manager
    query_vector = [0.1, 0.2, 0.3]
    mock_index.query.return_value.matches = [
        MagicMock(id="id1", score=0.9),
        MagicMock(id="id2", score=0.8)
    ]
    
    results = manager.search_similar(query_vector, top_k=2)
    
    assert results == [("id1", 0.9), ("id2", 0.8)]
    mock_index.query.assert_called_once_with(vector=query_vector, top_k=2, include_metadata=True)

def test_embed_and_upsert(pinecone_manager):
    manager, mock_index = pinecone_manager
    sentences = ["Hello world", "Test sentence"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    with patch.object(manager, 'model') as mock_model:
        mock_model.encode.return_value = embeddings
        ids = manager.embed_and_upsert(sentences)
        
        assert len(ids) == 2
        mock_index.upsert.assert_called_once()