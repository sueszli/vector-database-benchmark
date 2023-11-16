from copy import deepcopy
import pytest
import yaml
from awx.main.utils.safe_yaml import safe_dump

@pytest.mark.parametrize('value', [None, 1, 1.5, []])
def test_native_types(value):
    if False:
        i = 10
        return i + 15
    assert safe_dump(value) == yaml.safe_dump(value)

def test_empty():
    if False:
        while True:
            i = 10
    assert safe_dump({}) == ''

def test_raw_string():
    if False:
        i = 10
        return i + 15
    assert safe_dump('foo') == "!unsafe 'foo'\n"

def test_kv_null():
    if False:
        while True:
            i = 10
    assert safe_dump({'a': None}) == "!unsafe 'a': null\n"

def test_kv_null_safe():
    if False:
        i = 10
        return i + 15
    assert safe_dump({'a': None}, {'a': None}) == 'a: null\n'

def test_kv_null_unsafe():
    if False:
        i = 10
        return i + 15
    assert safe_dump({'a': ''}, {'a': None}) == "!unsafe 'a': !unsafe ''\n"

def test_kv_int():
    if False:
        i = 10
        return i + 15
    assert safe_dump({'a': 1}) == "!unsafe 'a': 1\n"

def test_kv_float():
    if False:
        while True:
            i = 10
    assert safe_dump({'a': 1.5}) == "!unsafe 'a': 1.5\n"

def test_kv_unsafe():
    if False:
        print('Hello World!')
    assert safe_dump({'a': 'b'}) == "!unsafe 'a': !unsafe 'b'\n"

def test_kv_unsafe_unicode():
    if False:
        for i in range(10):
            print('nop')
    assert safe_dump({'a': u'🐉'}) == '!unsafe \'a\': !unsafe "\\U0001F409"\n'

def test_kv_unsafe_in_list():
    if False:
        for i in range(10):
            print('nop')
    assert safe_dump({'a': ['b']}) == "!unsafe 'a':\n- !unsafe 'b'\n"

def test_kv_unsafe_in_mixed_list():
    if False:
        while True:
            i = 10
    assert safe_dump({'a': [1, 'b']}) == "!unsafe 'a':\n- 1\n- !unsafe 'b'\n"

def test_kv_unsafe_deep_nesting():
    if False:
        i = 10
        return i + 15
    yaml = safe_dump({'a': [1, [{'b': {'c': [{'d': 'e'}]}}]]})
    for x in ('a', 'b', 'c', 'd', 'e'):
        assert "!unsafe '{}'".format(x) in yaml

def test_kv_unsafe_multiple():
    if False:
        return 10
    assert safe_dump({'a': 'b', 'c': 'd'}) == '\n'.join(["!unsafe 'a': !unsafe 'b'", "!unsafe 'c': !unsafe 'd'", ''])

def test_safe_marking():
    if False:
        print('Hello World!')
    assert safe_dump({'a': 'b'}, safe_dict={'a': 'b'}) == 'a: b\n'

def test_safe_marking_mixed():
    if False:
        print('Hello World!')
    assert safe_dump({'a': 'b', 'c': 'd'}, safe_dict={'a': 'b'}) == '\n'.join(['a: b', "!unsafe 'c': !unsafe 'd'", ''])

def test_safe_marking_deep_nesting():
    if False:
        return 10
    deep = {'a': [1, [{'b': {'c': [{'d': 'e'}]}}]]}
    yaml = safe_dump(deep, deepcopy(deep))
    for x in ('a', 'b', 'c', 'd', 'e'):
        assert "!unsafe '{}'".format(x) not in yaml

def test_deep_diff_unsafe_marking():
    if False:
        while True:
            i = 10
    deep = {'a': [1, [{'b': {'c': [{'d': 'e'}]}}]]}
    jt_vars = deepcopy(deep)
    deep['a'][1][0]['b']['z'] = 'not safe'
    yaml = safe_dump(deep, jt_vars)
    assert "!unsafe 'z'" in yaml