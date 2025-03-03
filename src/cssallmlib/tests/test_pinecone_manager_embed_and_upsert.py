import pytest
from unittest.mock import patch
from cssallmlib.vectordb.pinecone_db import PineconeManager
import numpy as np


@pytest.fixture
def mock_pinecone():
    with patch("pinecone.Pinecone") as mock_init, patch("pinecone.Index") as mock_index:
        mock_instance = mock_index.return_value
        yield {"init": mock_init, "index": mock_instance}


@pytest.fixture
def mock_sentence_transformer():
    with patch("cssallmlib.vectordb.operations.SentenceTransformer") as mock_st:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        yield mock_instance


@pytest.fixture
def pinecone_manager(mock_pinecone, mock_sentence_transformer):
    return PineconeManager(
        api_key="test-key", environment="test-env", index_name="test-index"
    )


def test_embed_and_upsert(pinecone_manager, mock_sentence_transformer, mock_pinecone):
    sentences = ["test sentence 1", "test sentence 2"]

    ids = pinecone_manager.embed_and_upsert(sentences)

    assert len(ids) == 2
    mock_sentence_transformer.encode.assert_called_once_with(sentences)
    mock_pinecone["index"].upsert.assert_called_once()
