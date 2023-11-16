import pytest
from unittest.mock import patch
from llama_index.vector_stores import PineconeVectorStore, RedisVectorStore
from superagi.resource_manager.llama_vector_store_factory import LlamaVectorStoreFactory
from superagi.types.vector_store_types import VectorStoreType

def test_llama_vector_store_factory():
    if False:
        i = 10
        return i + 15
    vector_store_name = VectorStoreType.PINECONE
    index_name = 'test_index_name'
    factory = LlamaVectorStoreFactory(vector_store_name, index_name)
    with patch.object(PineconeVectorStore, '__init__', return_value=None):
        vector_store = factory.get_vector_store()
        assert isinstance(vector_store, PineconeVectorStore)
    factory.vector_store_name = VectorStoreType.REDIS
    with patch.object(RedisVectorStore, '__init__', return_value=None), patch('superagi.config.config.get_config', return_value=None):
        vector_store = factory.get_vector_store()
        assert isinstance(vector_store, RedisVectorStore)
    factory.vector_store_name = 'unknown'
    with pytest.raises(ValueError) as exc_info:
        factory.get_vector_store()
    assert str(exc_info.value) == 'unknown vector store is not supported yet.'