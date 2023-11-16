import hashlib
import os
from unittest.mock import Mock, patch
import pytest
from embedchain.loaders.notion import NotionLoader

@pytest.fixture
def notion_loader():
    if False:
        for i in range(10):
            print('nop')
    with patch.dict(os.environ, {'NOTION_INTEGRATION_TOKEN': 'test_notion_token'}):
        yield NotionLoader()

def test_load_data(notion_loader):
    if False:
        return 10
    source = 'https://www.notion.so/Test-Page-1234567890abcdef1234567890abcdef'
    mock_text = 'This is a test page.'
    expected_doc_id = hashlib.sha256((mock_text + source).encode()).hexdigest()
    expected_data = [{'content': mock_text, 'meta_data': {'url': 'notion-12345678-90ab-cdef-1234-567890abcdef'}}]
    mock_page = Mock()
    mock_page.text = mock_text
    mock_documents = [mock_page]
    with patch('embedchain.loaders.notion.NotionPageReader') as mock_reader:
        mock_reader.return_value.load_data.return_value = mock_documents
        result = notion_loader.load_data(source)
    assert result['doc_id'] == expected_doc_id
    assert result['data'] == expected_data