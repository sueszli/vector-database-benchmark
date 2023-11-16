import pytest
import numpy as np
from superagi.vector_store import qdrant
from superagi.vector_store.embedding.openai import OpenAiEmbedding
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient

@pytest.fixture
def client():
    if False:
        return 10
    client = QdrantClient(':memory:')
    yield client

@pytest.fixture
def mock_openai_embedding(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(OpenAiEmbedding, 'get_embedding', lambda self, text: np.random.random(3).tolist())

@pytest.fixture
def store(client, mock_openai_embedding):
    if False:
        return 10
    client.create_collection(collection_name='Test_collection', vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    yield qdrant.Qdrant(client, OpenAiEmbedding(api_key='test_api_key'), 'Test_collection')
    client.delete_collection('Test_collection')

def test_add_texts(store):
    if False:
        i = 10
        return i + 15
    car_companies = ['Rolls-Royce', 'Bentley', 'Ferrari', 'Lamborghini', 'Aston Martin', 'Porsche', 'Bugatti', 'Maserati', 'McLaren', 'Mercedes-Benz']
    assert len(store.add_texts(car_companies)) == len(car_companies)

def test_get_matching_text(store):
    if False:
        print('Hello World!')
    car_companies = ['Rolls-Royce', 'Bentley', 'Ferrari', 'Lamborghini', 'Aston Martin', 'Porsche', 'Bugatti', 'Maserati', 'McLaren', 'Mercedes-Benz']
    store.add_texts(car_companies)
    assert len(store.get_matching_text(k=2, text='McLaren')) == 2