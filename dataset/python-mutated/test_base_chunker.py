import hashlib
from unittest.mock import MagicMock
import pytest
from embedchain.chunkers.base_chunker import BaseChunker
from embedchain.models.data_type import DataType

@pytest.fixture
def text_splitter_mock():
    if False:
        print('Hello World!')
    return MagicMock()

@pytest.fixture
def loader_mock():
    if False:
        while True:
            i = 10
    return MagicMock()

@pytest.fixture
def app_id():
    if False:
        return 10
    return 'test_app'

@pytest.fixture
def data_type():
    if False:
        print('Hello World!')
    return DataType.TEXT

@pytest.fixture
def chunker(text_splitter_mock, data_type):
    if False:
        for i in range(10):
            print('nop')
    text_splitter = text_splitter_mock
    chunker = BaseChunker(text_splitter)
    chunker.set_data_type(data_type)
    return chunker

def test_create_chunks(chunker, text_splitter_mock, loader_mock, app_id, data_type):
    if False:
        return 10
    text_splitter_mock.split_text.return_value = ['Chunk 1', 'Chunk 2']
    loader_mock.load_data.return_value = {'data': [{'content': 'Content 1', 'meta_data': {'url': 'URL 1'}}], 'doc_id': 'DocID'}
    result = chunker.create_chunks(loader_mock, 'test_src', app_id)
    expected_ids = [f'{app_id}--' + hashlib.sha256(('Chunk 1' + 'URL 1').encode()).hexdigest(), f'{app_id}--' + hashlib.sha256(('Chunk 2' + 'URL 1').encode()).hexdigest()]
    assert result['documents'] == ['Chunk 1', 'Chunk 2']
    assert result['ids'] == expected_ids
    assert result['metadatas'] == [{'url': 'URL 1', 'data_type': data_type.value, 'doc_id': f'{app_id}--DocID'}, {'url': 'URL 1', 'data_type': data_type.value, 'doc_id': f'{app_id}--DocID'}]
    assert result['doc_id'] == f'{app_id}--DocID'

def test_get_chunks(chunker, text_splitter_mock):
    if False:
        print('Hello World!')
    text_splitter_mock.split_text.return_value = ['Chunk 1', 'Chunk 2']
    content = 'This is a test content.'
    result = chunker.get_chunks(content)
    assert len(result) == 2
    assert result == ['Chunk 1', 'Chunk 2']

def test_set_data_type(chunker):
    if False:
        for i in range(10):
            print('nop')
    chunker.set_data_type(DataType.MDX)
    assert chunker.data_type == DataType.MDX

def test_get_word_count(chunker):
    if False:
        i = 10
        return i + 15
    documents = ['This is a test.', 'Another test.']
    result = chunker.get_word_count(documents)
    assert result == 6