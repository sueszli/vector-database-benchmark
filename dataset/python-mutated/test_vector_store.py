"""
Tests regarding the vector store class, including checking
compatibility between different transformers and local vector
stores (index.faiss)
"""
import pytest
from application.vectorstore.faiss import FaissStore
from application.core.settings import settings

def test_init_local_faiss_store_huggingface():
    if False:
        print('Hello World!')
    '\n    Test that asserts that trying to initialize a FaissStore with\n    the huggingface sentence transformer below together with the\n    index.faiss file in the application/ folder results in a\n    dimension mismatch error.\n    '
    settings.EMBEDDINGS_NAME = 'huggingface_sentence-transformers/all-mpnet-base-v2'
    with pytest.raises(ValueError):
        FaissStore('application/', '', None)