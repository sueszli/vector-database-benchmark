import time
import pytest
from unittest import mock
import requests
from apprise.common import ConfigFormat
from apprise.config.ConfigHTTP import ConfigHTTP
from apprise.plugins.NotifyBase import NotifyBase
from apprise.common import NOTIFY_SCHEMA_MAP
import logging
logging.disable(logging.CRITICAL)
REQUEST_EXCEPTIONS = (requests.ConnectionError(0, 'requests.ConnectionError() not handled'), requests.RequestException(0, 'requests.RequestException() not handled'), requests.HTTPError(0, 'requests.HTTPError() not handled'), requests.ReadTimeout(0, 'requests.ReadTimeout() not handled'), requests.TooManyRedirects(0, 'requests.TooManyRedirects() not handled'))

@mock.patch('requests.post')
def test_config_http(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    API: ConfigHTTP() object\n\n    '

    class GoodNotification(NotifyBase):

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(*args, **kwargs)

        def notify(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return True

        def url(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return ''
    NOTIFY_SCHEMA_MAP['good'] = GoodNotification
    default_content = 'taga,tagb=good://server01'

    class DummyResponse:
        """
        A dummy response used to manage our object
        """
        status_code = requests.codes.ok
        headers = {'Content-Length': len(default_content), 'Content-Type': 'text/plain'}
        text = default_content
        ptr = None

        def close(self):
            if False:
                print('Hello World!')
            return

        def raise_for_status(self):
            if False:
                while True:
                    i = 10
            return

        def __enter__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self

        def __exit__(self, *args, **kwargs):
            if False:
                return 10
            return
    dummy_response = DummyResponse()
    mock_post.return_value = dummy_response
    assert ConfigHTTP.parse_url('garbage://') is None
    results = ConfigHTTP.parse_url('http://user:pass@localhost?+key=value')
    assert isinstance(results, dict)
    ch = ConfigHTTP(**results)
    assert isinstance(ch.url(), str) is True
    assert isinstance(ch.read(), str) is True
    assert len(ch) == 1
    results = ConfigHTTP.parse_url('http://localhost:8080/path/')
    assert isinstance(results, dict)
    ch = ConfigHTTP(**results)
    assert isinstance(ch.url(), str) is True
    assert isinstance(ch.read(), str) is True
    assert len(ch) == 1
    mock_post.reset_mock()
    results = ConfigHTTP.parse_url('http://localhost:8080/path/?cache=30')
    assert mock_post.call_count == 0
    assert isinstance(ch.url(), str) is True
    assert isinstance(results, dict)
    ch = ConfigHTTP(**results)
    assert mock_post.call_count == 0
    assert isinstance(ch.url(), str) is True
    assert mock_post.call_count == 0
    assert isinstance(ch.read(), str) is True
    assert mock_post.call_count == 1
    mock_post.reset_mock()
    assert ch.expired() is True
    assert ch
    assert mock_post.call_count == 1
    mock_post.reset_mock()
    assert ch.expired() is False
    assert len(ch) == 1
    assert mock_post.call_count == 0
    mock_post.reset_mock()
    assert ch
    assert len(ch.servers()) == 1
    assert len(ch) == 1
    assert mock_post.call_count == 0
    with mock.patch('time.time', return_value=time.time() + 10):
        assert ch.expired() is False
        assert ch
        assert len(ch.servers()) == 1
        assert len(ch) == 1
    assert mock_post.call_count == 0
    with mock.patch('time.time', return_value=time.time() + 31):
        assert ch.expired() is True
        assert ch
        assert len(ch.servers()) == 1
        assert len(ch) == 1
    assert mock_post.call_count == 1
    assert len(ch) == 1
    results = ConfigHTTP.parse_url('http://localhost:8080/path/?cache=False')
    assert isinstance(results, dict)
    assert isinstance(ch.url(), str) is True
    results = ConfigHTTP.parse_url('http://localhost:8080/path/?cache=-10')
    assert isinstance(results, dict)
    with pytest.raises(TypeError):
        ch = ConfigHTTP(**results)
    results = ConfigHTTP.parse_url('http://user@localhost?format=text')
    assert isinstance(results, dict)
    ch = ConfigHTTP(**results)
    assert isinstance(ch.url(), str) is True
    assert isinstance(ch.read(), str) is True
    assert len(ch) == 1
    results = ConfigHTTP.parse_url('https://localhost')
    assert isinstance(results, dict)
    ch = ConfigHTTP(**results)
    assert isinstance(ch.url(), str) is True
    assert isinstance(ch.read(), str) is True
    assert len(ch) == 1
    ch = ConfigHTTP(**results)
    ref = ch[0]
    assert isinstance(ref, NotifyBase) is True
    ref_popped = ch.pop(0)
    assert isinstance(ref_popped, NotifyBase) is True
    assert ref == ref_popped
    assert len(ch) == 0
    ch = ConfigHTTP(**results)
    assert isinstance(ch.pop(0), NotifyBase) is True
    ch = ConfigHTTP(**results)
    assert isinstance(ch[0], NotifyBase) is True
    assert isinstance(ch[0], NotifyBase) is True
    ch = ConfigHTTP(**results)
    iter(ch)
    iter(ch)
    ch.max_buffer_size = len(dummy_response.text)
    assert isinstance(ch.read(), str) is True
    yaml_supported_types = ('text/yaml', 'text/x-yaml', 'application/yaml', 'application/x-yaml')
    for st in yaml_supported_types:
        dummy_response.headers['Content-Type'] = st
        ch.default_config_format = None
        assert isinstance(ch.read(), str) is True
        assert ch.default_config_format == ConfigFormat.YAML
    text_supported_types = ('text/plain', 'text/html')
    for st in text_supported_types:
        dummy_response.headers['Content-Type'] = st
        ch.default_config_format = None
        assert isinstance(ch.read(), str) is True
        assert ch.default_config_format == ConfigFormat.TEXT
    ukwn_supported_types = ('text/css', 'application/zip')
    for st in ukwn_supported_types:
        dummy_response.headers['Content-Type'] = st
        ch.default_config_format = None
        assert isinstance(ch.read(), str) is True
        assert ch.default_config_format is None
    del dummy_response.headers['Content-Type']
    ch.default_config_format = None
    assert isinstance(ch.read(), str) is True
    assert ch.default_config_format is None
    dummy_response.headers['Content-Type'] = 'text/plain'
    max_buffer_size = ch.max_buffer_size
    ch.max_buffer_size = len(dummy_response.text) - 1
    assert ch.read() is None
    ch.max_buffer_size = max_buffer_size
    dummy_response.headers['Content-Length'] = 'garbage'
    assert isinstance(ch.read(), str) is True
    dummy_response.headers['Content-Length'] = 'None'
    assert isinstance(ch.read(), str) is True
    dummy_response.text = 'a' * ch.max_buffer_size
    assert isinstance(ch.read(), str) is True
    dummy_response.text = 'b' * (ch.max_buffer_size + 1)
    assert ch.read() is None
    dummy_response.status_code = 400
    assert ch.read() is None
    ch.max_error_buffer_size = 0
    assert ch.read() is None
    for _exception in REQUEST_EXCEPTIONS:
        mock_post.side_effect = _exception
        assert ch.read() is None
    ch.max_buffer_size = max_buffer_size