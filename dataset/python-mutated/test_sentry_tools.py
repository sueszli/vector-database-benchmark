import pytest
from tribler.core.sentry_reporter.sentry_tools import _re_search_exception, delete_item, distinct_by, extract_dict, format_version, get_first_item, get_last_item, get_value, modify_value, obfuscate_string, parse_last_core_output

def test_first():
    if False:
        while True:
            i = 10
    assert get_first_item(None, '') == ''
    assert get_first_item([], '') == ''
    assert get_first_item(['some'], '') == 'some'
    assert get_first_item(['some', 'value'], '') == 'some'
    assert get_first_item((), '') == ''
    assert get_first_item(('some', 'value'), '') == 'some'
    assert get_first_item(None, None) is None

def test_last():
    if False:
        for i in range(10):
            print('nop')
    assert get_last_item(None, '') == ''
    assert get_last_item([], '') == ''
    assert get_last_item(['some'], '') == 'some'
    assert get_last_item(['some', 'value'], '') == 'value'
    assert get_last_item((), '') == ''
    assert get_last_item(('some', 'value'), '') == 'value'
    assert get_last_item(None, None) is None

def test_delete():
    if False:
        print('Hello World!')
    assert delete_item({}, None) == {}
    assert delete_item({'key': 'value'}, None) == {'key': 'value'}
    assert delete_item({'key': 'value'}, 'missed_key') == {'key': 'value'}
    assert delete_item({'key': 'value'}, 'key') == {}

def test_modify():
    if False:
        print('Hello World!')
    assert modify_value(None, None, None) is None
    assert modify_value({}, None, None) == {}
    assert modify_value({}, '', None) == {}
    assert modify_value({}, 'key', lambda value: '') == {}
    assert modify_value({'a': 'b'}, 'key', lambda value: '') == {'a': 'b'}
    assert modify_value({'a': 'b', 'key': 'value'}, 'key', lambda value: '') == {'a': 'b', 'key': ''}

def test_safe_get():
    if False:
        print('Hello World!')
    assert get_value(None, None, None) is None
    assert get_value(None, None, {}) == {}
    assert get_value(None, 'key', {}) == {}
    assert get_value({'key': 'value'}, 'key', {}) == 'value'
    assert get_value({'key': 'value'}, 'key1', {}) == {}

def test_distinct():
    if False:
        for i in range(10):
            print('nop')
    assert distinct_by(None, None) is None
    assert distinct_by([], None) == []
    assert distinct_by([{'key': 'b'}, {'key': 'b'}, {'key': 'c'}, {'': ''}], 'key') == [{'key': 'b'}, {'key': 'c'}, {'': ''}]
    assert distinct_by([{'a': {}}], 'b') == [{'a': {}}]
FORMATTED_VERSIONS = [(None, None), ('', ''), ('7.6.0', '7.6.0'), ('7.6.0-GIT', 'dev'), ('7.7.1-17-gcb73f7baa', '7.7.1'), ('7.7.1-RC1-10-abcd', '7.7.1-RC1'), ('7.7.1-exp1-1-abcd ', '7.7.1-exp1'), ('7.7.1-someresearchtopic-7-abcd ', '7.7.1-someresearchtopic')]

@pytest.mark.parametrize('git_version, sentry_version', FORMATTED_VERSIONS)
def test_format_version(git_version, sentry_version):
    if False:
        print('Hello World!')
    assert format_version(git_version) == sentry_version

def test_extract_dict():
    if False:
        return 10
    assert not extract_dict(None, None)
    assert extract_dict({}, '') == {}
    assert extract_dict({'k': 'v', 'k1': 'v1'}, '\\w$') == {'k': 'v'}
OBFUSCATED_STRINGS = [(None, None), ('', ''), ('any', 'challenge'), ('string', 'quality')]
EXCEPTION_STRINGS = [('OverflowError: bind(): port must be 0-65535', ('OverflowError', 'bind(): port must be 0-65535')), ("pony.orm.core.TransactionIntegrityError : MiscData['db_version'] cannot be stored. IntegrityError: UNIQUE", ('pony.orm.core.TransactionIntegrityError', "MiscData['db_version'] cannot be stored. IntegrityError: UNIQUE")), ('ERROR <exception_handler:100>', None)]

@pytest.mark.parametrize('given, expected', OBFUSCATED_STRINGS)
def test_obfuscate_string(given, expected):
    if False:
        for i in range(10):
            print('nop')
    assert obfuscate_string(given) == expected

@pytest.mark.parametrize('given, expected', EXCEPTION_STRINGS)
def test_parse_last_core_output_re(given, expected):
    if False:
        while True:
            i = 10
    if (m := _re_search_exception.match(given)):
        (exception_type, exception_text) = expected
        assert m.group(1) == exception_type
        assert m.group(2) == exception_text
    else:
        assert m == expected

def test_parse_last_core_output():
    if False:
        for i in range(10):
            print('nop')
    last_core_output = '\npony.orm.core.TransactionIntegrityError: Object MiscData[\'db_version\'] cannot be stored in the database. IntegrityError\nERROR <exception_handler:100> CoreExceptionHandler.unhandled_error_observer(): Unhandled exception occurred! bind(): \nTraceback (most recent call last):\n  File "/Users/<user>/Projects/github.com/Tribler/tribler/src/tribler/core/components/component.py", line 61, in start\n    await self.run()\n  File "/Users/<user>/Projects/github.com/Tribler/tribler/src/tribler/core/components/restapi/restapi_component.py", \n    await rest_manager.start()\n  File "/Users/<user>/Projects/github.com/Tribler/tribler/src/tribler/core/components/restapi/rest/rest_manager.py", \n    await self.site.start()\n  File "/Users/<user>/Projects/github.com/Tribler/tribler/venv/lib/python3.8/site-packages/aiohttp/web_runner.py", \n    self._server = await loop.create_server(\n  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.8/lib/python3.8/\n    sock.bind(sa)\nOverflowError: bind(): port must be 0-65535.Sentry is attempting to send 1 pending error messages\nWaiting up to 2 seconds\nPress Ctrl-C to quit\n    '
    last_core_exception = parse_last_core_output(last_core_output)
    assert last_core_exception.type == 'OverflowError'
    assert last_core_exception.message == 'bind(): port must be 0-65535.'

def test_parse_last_core_output_no_match():
    if False:
        while True:
            i = 10
    last_core_exception = parse_last_core_output('last core output without exceptions')
    assert not last_core_exception