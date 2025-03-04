import pytest
import numpy as np
from unittest.mock import Mock, patch
from cssallmlib.vectordb.pinecone_db import PineconeManager


@pytest.fixture
def mock_pinecone():
    with patch("cssallmlib.vectordb.pinecone_db.pinecone") as mock_pc:
        # Mock the Pinecone client and index
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pc.Pinecone.return_value = mock_pc
        yield mock_pc


@pytest.fixture
def pinecone_manager(mock_pinecone):
    return PineconeManager(
        api_key="test_key", environment="test_env", index_name="test_index"
    )


def test_init(pinecone_manager, mock_pinecone):
    """Test initialization of PineconeManager"""
    assert pinecone_manager.index is not None
    mock_pinecone.Pinecone.assert_called_once_with(
        api_key="test_key", environment="test_env"
    )


def test_upsert_vectors(pinecone_manager):
    """Test upserting vectors"""
    # Test data
    vectors = [("id1", [0.1, 0.2, 0.3]), ("id2", [0.4, 0.5, 0.6])]
    metadata = {"id1": {"text": "test1"}, "id2": {"text": "test2"}}

    # Call the method
    pinecone_manager.upsert_vectors(vectors, metadata)

    # Verify the index.upsert was called with correct parameters
    pinecone_manager.index.upsert.assert_called_once()
    call_args = pinecone_manager.index.upsert.call_args[1]
    assert len(call_args["vectors"]) == 2
    assert call_args["vectors"][0][0] == "id1"
    assert call_args["vectors"][0][2] == {"text": "test1"}


def test_upsert_vectors_invalid_input(pinecone_manager):
    """Test upsert_vectors with invalid inputs"""
    # Test with non-list input
    with pytest.raises(ValueError, match="vectors should be a list of tuples"):
        pinecone_manager.upsert_vectors("not_a_list")

    # Test with invalid tuple format
    with pytest.raises(
        ValueError, match="vectors should be a list of tuples of length 2"
    ):
        pinecone_manager.upsert_vectors([("id1", "vec1", "extra")])

    # Test with invalid metadata type
    with pytest.raises(ValueError, match="metadata should be a dictionary"):
        pinecone_manager.upsert_vectors([("id1", [0.1, 0.2])], metadata="not_a_dict")


def test_search_similar(pinecone_manager):
    """Test searching for similar vectors"""
    # Mock the query response
    mock_matches = [Mock(id="id1", score=0.95), Mock(id="id2", score=0.85)]
    pinecone_manager.index.query.return_value.matches = mock_matches

    # Test data
    query_vector = [0.1, 0.2, 0.3]
    top_k = 2

    # Call the method
    results = pinecone_manager.search_similar(query_vector, top_k)

    # Verify results
    assert len(results) == 2
    assert results[0] == ("id1", 0.95)
    assert results[1] == ("id2", 0.85)

    # Verify the query was called with correct parameters
    pinecone_manager.index.query.assert_called_once_with(
        vector=query_vector, top_k=top_k, include_metadata=True
    )


def test_embed_and_upsert(pinecone_manager):
    """Test embedding and upserting sentences"""
    # Mock the model's encode method
    pinecone_manager.model = Mock()
    pinecone_manager.model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    # Test data
    sentences = ["test1", "test2"]
    metadata = [{"text": "test1"}, {"text": "test2"}]

    # Call the method
    ids = pinecone_manager.embed_and_upsert(sentences, metadata=metadata)

    # Verify results
    assert len(ids) == 2
    assert all(isinstance(id_, str) for id_ in ids)

    # Verify the model.encode was called
    pinecone_manager.model.encode.assert_called_once_with(sentences)


def test_embed_and_upsert_invalid_input(pinecone_manager):
    """Test embed_and_upsert with invalid inputs"""
    # Mock the model's encode method
    pinecone_manager.model = Mock()
    pinecone_manager.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    # Test with non-list sentences
    with pytest.raises(ValueError, match="sentences should be a list of strings"):
        pinecone_manager.embed_and_upsert("not_a_list")

    # Test with invalid metadata length
    with pytest.raises(ValueError):
        pinecone_manager.embed_and_upsert(
            ["test1", "test2"],
            metadata=[{"text": "test1"}],  # Only one metadata item for two sentences
        )
