from __future__ import annotations
from ansible.module_utils.urls import generic_urlparse
from urllib.parse import urlparse, urlunparse

def test_generic_urlparse():
    if False:
        i = 10
        return i + 15
    url = 'https://ansible.com/blog'
    parts = urlparse(url)
    generic_parts = generic_urlparse(parts)
    assert generic_parts.as_list() == list(parts)
    assert urlunparse(generic_parts.as_list()) == url

def test_generic_urlparse_netloc():
    if False:
        for i in range(10):
            print('nop')
    url = 'https://ansible.com:443/blog'
    parts = urlparse(url)
    generic_parts = generic_urlparse(parts)
    assert generic_parts.hostname == parts.hostname
    assert generic_parts.hostname == 'ansible.com'
    assert generic_parts.port == 443
    assert urlunparse(generic_parts.as_list()) == url