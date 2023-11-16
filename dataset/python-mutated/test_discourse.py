import pytest
import requests
from embedchain.loaders.discourse import DiscourseLoader

@pytest.fixture
def discourse_loader_config():
    if False:
        print('Hello World!')
    return {'domain': 'https://example.com/'}

@pytest.fixture
def discourse_loader(discourse_loader_config):
    if False:
        for i in range(10):
            print('nop')
    return DiscourseLoader(config=discourse_loader_config)

def test_discourse_loader_init_with_valid_config():
    if False:
        print('Hello World!')
    config = {'domain': 'https://example.com/'}
    loader = DiscourseLoader(config=config)
    assert loader.domain == 'https://example.com/'

def test_discourse_loader_init_with_missing_config():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='DiscourseLoader requires a config'):
        DiscourseLoader()

def test_discourse_loader_init_with_missing_domain():
    if False:
        print('Hello World!')
    config = {'another_key': 'value'}
    with pytest.raises(ValueError, match='DiscourseLoader requires a domain'):
        DiscourseLoader(config=config)

def test_discourse_loader_check_query_with_valid_query(discourse_loader):
    if False:
        return 10
    discourse_loader._check_query('sample query')

def test_discourse_loader_check_query_with_empty_query(discourse_loader):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='DiscourseLoader requires a query'):
        discourse_loader._check_query('')

def test_discourse_loader_check_query_with_invalid_query_type(discourse_loader):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='DiscourseLoader requires a query'):
        discourse_loader._check_query(123)

def test_discourse_loader_load_post_with_valid_post_id(discourse_loader, monkeypatch):
    if False:
        print('Hello World!')

    def mock_get(*args, **kwargs):
        if False:
            while True:
                i = 10

        class MockResponse:

            def json(self):
                if False:
                    return 10
                return {'raw': 'Sample post content'}

            def raise_for_status(self):
                if False:
                    return 10
                pass
        return MockResponse()
    monkeypatch.setattr(requests, 'get', mock_get)
    post_data = discourse_loader._load_post(123)
    assert post_data['content'] == 'Sample post content'
    assert 'meta_data' in post_data

def test_discourse_loader_load_post_with_invalid_post_id(discourse_loader, monkeypatch, caplog):
    if False:
        for i in range(10):
            print('nop')

    def mock_get(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        class MockResponse:

            def raise_for_status(self):
                if False:
                    return 10
                raise requests.exceptions.RequestException('Test error')
        return MockResponse()
    monkeypatch.setattr(requests, 'get', mock_get)
    discourse_loader._load_post(123)
    assert 'Failed to load post' in caplog.text

def test_discourse_loader_load_data_with_valid_query(discourse_loader, monkeypatch):
    if False:
        i = 10
        return i + 15

    def mock_get(*args, **kwargs):
        if False:
            print('Hello World!')

        class MockResponse:

            def json(self):
                if False:
                    print('Hello World!')
                return {'grouped_search_result': {'post_ids': [123, 456, 789]}}

            def raise_for_status(self):
                if False:
                    print('Hello World!')
                pass
        return MockResponse()
    monkeypatch.setattr(requests, 'get', mock_get)

    def mock_load_post(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return {'content': 'Sample post content', 'meta_data': {'url': 'https://example.com/posts/123.json', 'created_at': '2021-01-01', 'username': 'test_user', 'topic_slug': 'test_topic', 'score': 10}}
    monkeypatch.setattr(discourse_loader, '_load_post', mock_load_post)
    data = discourse_loader.load_data('sample query')
    assert len(data['data']) == 3
    assert data['data'][0]['content'] == 'Sample post content'
    assert data['data'][0]['meta_data']['url'] == 'https://example.com/posts/123.json'
    assert data['data'][0]['meta_data']['created_at'] == '2021-01-01'
    assert data['data'][0]['meta_data']['username'] == 'test_user'
    assert data['data'][0]['meta_data']['topic_slug'] == 'test_topic'
    assert data['data'][0]['meta_data']['score'] == 10