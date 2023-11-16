from __future__ import annotations
from typing import Any
from sentry.constants import MAX_CULPRIT_LENGTH
from sentry.culprit import generate_culprit
from sentry.grouping.utils import hash_from_values

def test_with_exception_interface():
    if False:
        for i in range(10):
            print('nop')
    data = {'exception': {'values': [{'stacktrace': {'frames': [{'lineno': 1, 'filename': 'foo.py'}, {'lineno': 1, 'filename': 'bar.py', 'in_app': True}]}}]}, 'stacktrace': {'frames': [{'lineno': 1, 'filename': 'NOTME.py'}, {'lineno': 1, 'filename': 'PLZNOTME.py', 'in_app': True}]}, 'request': {'url': 'http://example.com'}}
    assert generate_culprit(data) == 'bar.py in ?'

def test_with_missing_exception_stacktrace():
    if False:
        for i in range(10):
            print('nop')
    data = {'exception': {'values': [{'stacktrace': None}, {'stacktrace': {'frames': None}}, {'stacktrace': {'frames': [None]}}]}, 'request': {'url': 'http://example.com'}}
    assert generate_culprit(data) == 'http://example.com'

def test_with_stacktrace_interface():
    if False:
        for i in range(10):
            print('nop')
    data = {'stacktrace': {'frames': [{'lineno': 1, 'filename': 'NOTME.py'}, {'lineno': 1, 'filename': 'PLZNOTME.py', 'in_app': True}]}, 'request': {'url': 'http://example.com'}}
    assert generate_culprit(data) == 'PLZNOTME.py in ?'

def test_with_missing_stacktrace_frames():
    if False:
        return 10
    data = {'stacktrace': {'frames': None}, 'request': {'url': 'http://example.com'}}
    assert generate_culprit(data) == 'http://example.com'

def test_with_empty_stacktrace():
    if False:
        while True:
            i = 10
    data = {'stacktrace': None, 'request': {'url': 'http://example.com'}}
    assert generate_culprit(data) == 'http://example.com'

def test_with_only_http_interface():
    if False:
        while True:
            i = 10
    data: dict[str, Any] = {'request': {'url': 'http://example.com'}}
    assert generate_culprit(data) == 'http://example.com'
    data = {'request': {'url': None}}
    assert generate_culprit(data) == ''
    data = {'request': {}}
    assert generate_culprit(data) == ''
    data = {'request': None}
    assert generate_culprit(data) == ''

def test_empty_data():
    if False:
        print('Hello World!')
    assert generate_culprit({}) == ''

def test_truncation():
    if False:
        for i in range(10):
            print('nop')
    data: dict[str, dict[str, Any]] = {'exception': {'values': [{'stacktrace': {'frames': [{'filename': 'x' * (MAX_CULPRIT_LENGTH + 1)}]}}]}}
    assert len(generate_culprit(data)) == MAX_CULPRIT_LENGTH
    data = {'stacktrace': {'frames': [{'filename': 'x' * (MAX_CULPRIT_LENGTH + 1)}]}}
    assert len(generate_culprit(data)) == MAX_CULPRIT_LENGTH
    data = {'request': {'url': 'x' * (MAX_CULPRIT_LENGTH + 1)}}
    assert len(generate_culprit(data)) == MAX_CULPRIT_LENGTH

def test_hash_from_values():
    if False:
        return 10
    result = hash_from_values(['foo', 'bar', 'foô'])
    assert result == '6d81588029ed4190110b2779ba952a00'

def test_nel_culprit():
    if False:
        print('Hello World!')
    data = {'type': 'nel', 'contexts': {'nel': {'phase': 'application', 'error_type': 'http.error'}, 'response': {'status_code': 418}}}
    assert generate_culprit(data) == 'The user agent successfully received a response, but it had a 418 status code'
    data = {'type': 'nel', 'contexts': {'nel': {'phase': 'connection', 'error_type': 'tcp.reset'}}}
    assert generate_culprit(data) == 'The TCP connection was reset'
    data = {'type': 'nel', 'contexts': {'nel': {'phase': 'dns', 'error_type': 'dns.weird'}}}
    assert generate_culprit(data) == 'dns.weird'
    data = {'type': 'nel', 'contexts': {'nel': {'phase': 'dns'}}}
    assert generate_culprit(data) == '<missing>'