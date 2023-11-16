import pytest
from azure.core.rest import HttpRequest

def _format_query_into_url(url, params):
    if False:
        while True:
            i = 10
    request = HttpRequest(method='GET', url=url, params=params)
    return request.url

def test_request_url_with_params():
    if False:
        return 10
    url = _format_query_into_url(url='a/b/c?t=y', params={'g': 'h'})
    assert url in ['a/b/c?g=h&t=y', 'a/b/c?t=y&g=h']

def test_request_url_with_params_as_list():
    if False:
        i = 10
        return i + 15
    url = _format_query_into_url(url='a/b/c?t=y', params={'g': ['h', 'i']})
    assert url in ['a/b/c?g=h&g=i&t=y', 'a/b/c?t=y&g=h&g=i']

def test_request_url_with_params_with_none_in_list():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        _format_query_into_url(url='a/b/c?t=y', params={'g': ['h', None]})

def test_request_url_with_params_with_none():
    if False:
        return 10
    with pytest.raises(ValueError):
        _format_query_into_url(url='a/b/c?t=y', params={'g': None})