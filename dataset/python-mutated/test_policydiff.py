"""
.. module: security_monkey.tests.core.test_policydiff
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Patrick Kelley <patrick@netflix.com> @monkeysecurity

"""
from security_monkey.common.PolicyDiff import PolicyDiff
TEST_CASES = [dict(old='{}', new='\n        {\n            "user_name": "test",\n            "must_change_password": "false",\n            "create_date": "2013-09-12T18:28:21Z"\n        }\n        ', expected_result='{<br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;"user_name": "test",</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;"must_change_password": "false",</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;"create_date": "2013-09-12T18:28:21Z"</font><br/>\n}<br/>\n'), dict(old='\n        {\n            "create_date": "2013-09-12T18:28:21Z",\n            "must_change_password": "false",\n            "user_name": "test"\n        }\n        ', new='\n        {\n            "create_date": "2013-09-12T18:28:21Z",\n            "must_change_password": "false",\n            "user_name": "test"\n        }\n        ', expected_result='{<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"create_date": "2013-09-12T18:28:21Z",</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"must_change_password": "false",</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"user_name": "test"</font><br/>\n}<br/>\n'), dict(old={'thelist': [{'rule': 'asdf'}], 'must_change_password': 'false', 'user_name': 'test'}, new={'thelist': [{'rule': 'asdf'}, {'rule': 'defg'}], 'must_change_password': 'false', 'user_name': 'test'}, expected_result='{<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"thelist": [<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"rule": "asdf"</font><br/>\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;},</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{<br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"rule": "defg"</font><br/>\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}</font><br/>\n&nbsp;&nbsp;&nbsp;&nbsp;],</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"must_change_password": "false",</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"user_name": "test"</font><br/>\n}<br/>\n')]

def test_produce():
    if False:
        print('Hello World!')
    for case in TEST_CASES:
        differ = PolicyDiff(case['new'], case['old'])
        html = differ.produceDiffHTML()
        if html != case['expected_result']:
            print(html)
        assert html == case['expected_result']
    differ = PolicyDiff({}, {})
    result = differ.produceDiffHTML()
    assert result == 'No Policy.<br/>'
    differ._old_policy = None
    differ._new_policy = None
    try:
        differ.produceDiffHTML()
        assert False
    except ValueError:
        pass
    differ._old_policy = []
    differ._new_policy = {}
    try:
        differ.produceDiffHTML()
        assert False
    except ValueError:
        pass
    differ._old_policy = 'old_policy'
    differ._new_policy = 'new_policy'
    result = differ.produceDiffHTML()
    assert result == "<font color='red'>old_policy</font><br/>\n<br/><font color='green'>new_policy</font><br/>\n"
    differ._old_policy = [1, 2, 3]
    differ._new_policy = [1, 2, 3]
    differ.produceDiffHTML()
    differ._old_policy = set([1, 2, 3])
    differ._new_policy = set([1, 2, 3])
    try:
        differ.produceDiffHTML()
        assert False
    except ValueError:
        pass

def test_form_brackets():
    if False:
        while True:
            i = 10
    from security_monkey.common.PolicyDiff import form_brackets
    test_values = [{'value': 'a_string', 'open': '"', 'close': '"'}, {'value': {'key': 'dictionary'}, 'open': '{<br/>\n', 'close': '}'}, {'value': [1, 2, 3], 'open': '[<br/>\n', 'close': ']'}, {'value': 123, 'open': '', 'close': ''}]
    for value in test_values:
        result = form_brackets(value['value'], 0)
        assert value['open'] == result['open']
        assert value['close'] == result['close']

def test_get_brackets():
    if False:
        print('Hello World!')
    from security_monkey.common.PolicyDiff import get_brackets
    values = [('str', dict(open='"', close='"')), ('unicode', dict(open='"', close='"')), ([1, 2, 3], dict(open='[', close=']')), ({'a': 123}, dict(open='{', close='}')), (True, dict(open='', close='')), (123, dict(open='', close=''))]
    for value in values:
        assert get_brackets(value[0]) == value[1]

def test_added():
    if False:
        for i in range(10):
            print('nop')
    from security_monkey.common.PolicyDiff import added
    assert added('asdf') == "<font color='green'>asdf</font><br/>\n"

def test_deleted():
    if False:
        i = 10
        return i + 15
    from security_monkey.common.PolicyDiff import deleted
    assert deleted('asdf') == "<font color='red'>asdf</font><br/>\n"

def test_same():
    if False:
        for i in range(10):
            print('nop')
    from security_monkey.common.PolicyDiff import same
    assert same('asdf') == "<font color='black'>asdf</font><br/>\n"

def test_str_distance():
    if False:
        i = 10
        return i + 15
    from security_monkey.common.PolicyDiff import str_distance
    values = [('abcdefg', 'abcdefg', 0), ('abcdefg', 'abcdef0', 1), ('axxxxfg', 'abcdefg', 4), ('axxxxfg123', 'abcdefg', 7)]
    for value in values:
        assert str_distance(value[0], value[1]) == value[2]

def test_find_most_similar():
    if False:
        while True:
            i = 10
    from security_monkey.common.PolicyDiff import find_most_similar
    values = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ', 1234567890, 'November 2, 1962']
    assert find_most_similar('ABCDEFGHIJKLMNOPQRSTU', values) == values[0]
    assert find_most_similar(123456789, values) == values[1]
    assert find_most_similar(1234567890, values) == values[1]
    assert find_most_similar('November', values) == values[2]
    values = ['Incredible']
    assert find_most_similar('November', values) == values[0]

def test_print_item():
    if False:
        i = 10
        return i + 15
    from security_monkey.common.PolicyDiff import print_item
    values = [('<script>', '&lt;script&gt;'), (123, '123'), (932.121, '932.121'), (None, 'null'), (True, 'true'), (False, 'false'), ({'1': '2'}, '<font color=\'black\'>"1": "2"</font><br/>\n'), (['1', '2'], '<font color=\'black\'>"1",</font><br/>\n<font color=\'black\'>"2"</font><br/>\n'), (set([1, 2, 3]), '')]
    for value in values:
        assert print_item(value[0], 'same', 0) == value[1]

def test_print_list():
    if False:
        return 10
    from security_monkey.common.PolicyDiff import print_list
    values = ['string', {'a': 'b'}, ['a', 'b', 'c'], [1, 2, 3], True, False, None, set(['not supported type'])]
    expected = '<font color=\'{color}\'>"string",</font><br/>\n<font color=\'{color}\'>[[<br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": "b"</font><br/>\n]],</font><br/>\n<font color=\'{color}\'>[<br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;"a",</font><br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;"b",</font><br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;"c"</font><br/>\n],</font><br/>\n<font color=\'{color}\'>[<br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;3</font><br/>\n],</font><br/>\n<font color=\'{color}\'>true,</font><br/>\n<font color=\'{color}\'>false,</font><br/>\n<font color=\'{color}\'>null,</font><br/>\n<font color=\'{color}\'></font><br/>\n'
    assert print_list(values, 'same', 0) == expected.format(color='black').replace('[[', '{').replace(']]', '}')
    assert print_list(values, 'deleted', 0) == expected.format(color='red').replace('[[', '{').replace(']]', '}')
    assert print_list(values, 'added', 0) == expected.format(color='green').replace('[[', '{').replace(']]', '}')

def test_print_dict():
    if False:
        for i in range(10):
            print('nop')
    from security_monkey.common.PolicyDiff import print_dict
    values = {'a': '<script>', 'b': True, 'c': None, 'd': {'da': 1}, 'e': [1, 2, 3], 'f': set([1, 2, 3])}
    expected = '<font color=\'{color}\'>"a": "&lt;script&gt;",</font><br/>\n<font color=\'{color}\'>"b": true,</font><br/>\n<font color=\'{color}\'>"c": null,</font><br/>\n<font color=\'{color}\'>"d": [[<br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;"da": 1</font><br/>\n]],</font><br/>\n<font color=\'{color}\'>"e": [<br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'{color}\'>&nbsp;&nbsp;&nbsp;&nbsp;3</font><br/>\n],</font><br/>\n<font color=\'{color}\'>"f": </font><br/>\n'
    assert print_dict(values, 'same', 0) == expected.format(color='black').replace('[[', '{').replace(']]', '}')
    assert print_dict(values, 'deleted', 0) == expected.format(color='red').replace('[[', '{').replace(']]', '}')
    assert print_dict(values, 'added', 0) == expected.format(color='green').replace('[[', '{').replace(']]', '}')

def test_sub_dict():
    if False:
        return 10
    from security_monkey.common.PolicyDiff import process_sub_dict
    values = [dict(a='hello', b='hello', x='<font color=\'black\'>"somekey": "hello",</font><br/>\n'), dict(a='hello', b='different', x='<font color=\'red\'>"somekey": "different",</font><br/>\n<font color=\'green\'>"somekey": "hello",</font><br/>\n'), dict(a=123, b=123, x='<font color=\'black\'>"somekey": 123,</font><br/>\n'), dict(a=123, b=1234, x='<font color=\'red\'>"somekey": 1234,</font><br/>\n<font color=\'green\'>"somekey": 123,</font><br/>\n'), dict(a={'a': 123}, b={'a': 123}, x='<font color=\'black\'>"somekey": {<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": 123</font><br/>\n},</font><br/>\n'), dict(a={'a': 123}, b={'a': 1234}, x='<font color=\'black\'>"somekey": {<br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": 1234,</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": 123</font><br/>\n},</font><br/>\n'), dict(a=[1, 2, 3, 4], b=[1, 2, 3, 4], x='<font color=\'black\'>"somekey": [<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;3,</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;4</font><br/>\n],</font><br/>\n'), dict(a=[1, 2, 3, 4], b=[1, 2, 3, 4, 5], x='<font color=\'black\'>"somekey": [<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;3,</font><br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;4,</font><br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;5</font><br/>\n],</font><br/>\n'), dict(a=set([1, 2, 3]), b=set([1, 2, 3]), x='')]
    for value in values:
        result = process_sub_dict('somekey', value['a'], value['b'], 0)
        if result != value['x']:
            print(('RE', result))
            print(('EX', value['x']))
        assert result == value['x']
    try:
        process_sub_dict('somenkey', 'a_str', ['a list'], 0)
        assert False
    except ValueError as e:
        pass

def test_constructor():
    if False:
        while True:
            i = 10
    from security_monkey.common.PolicyDiff import PolicyDiff
    try:
        PolicyDiff('{badjson}', None)
        assert False
    except ValueError:
        pass
    try:
        PolicyDiff(None, '{badjson}')
        assert False
    except ValueError:
        pass
    try:
        PolicyDiff({}, [])
        assert False
    except ValueError:
        pass
    import collections
    PolicyDiff(collections.defaultdict(), collections.defaultdict())

def test_diff_list():
    if False:
        while True:
            i = 10
    from security_monkey.common.PolicyDiff import diff_list
    values = [dict(a=['1', '2', 3, 3.0, True, False, None, dict(a='123'), ['list'], set([1, 2, 3])], b=['1', '2', 3, 3.0, True, False, None, dict(a='123'), ['list'], set([1, 2, 3])], x='<font color=\'black\'>"1",</font><br/>\n<font color=\'black\'>"2",</font><br/>\n<font color=\'black\'>3,</font><br/>\n<font color=\'black\'>3.0,</font><br/>\n<font color=\'black\'>true,</font><br/>\n<font color=\'black\'>false,</font><br/>\n<font color=\'black\'>null,</font><br/>\n<font color=\'black\'>{<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": "123"</font><br/>\n},</font><br/>\n<font color=\'black\'>[<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"list"</font><br/>\n]</font><br/>\n'), dict(a=[1, 2, 3], b=[1, 3, 4], x="<font color='black'>1,</font><br/>\n<font color='black'>3,</font><br/>\n<font color='red'>4,</font><br/>\n<font color='green'>2</font><br/>\n"), dict(a=['str', True, [1, 3], set([1, 2])], b=[], x='<font color=\'green\'>"str",</font><br/>\n<font color=\'green\'>true,</font><br/>\n<font color=\'green\'>[<br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;3</font><br/>\n]</font><br/>\n'), dict(a=[True], b=[False], x="<font color='red'>false,</font><br/>\n<font color='green'>true</font><br/>\n"), dict(a=[[1, 2, 3, 4, 5]], b=[[1, 2, 3, 4, 4]], x="<font color='black'>[<br/>\n<font color='black'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color='black'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color='black'>&nbsp;&nbsp;&nbsp;&nbsp;3,</font><br/>\n<font color='black'>&nbsp;&nbsp;&nbsp;&nbsp;4,</font><br/>\n<font color='green'>&nbsp;&nbsp;&nbsp;&nbsp;5</font><br/>\n]</font><br/>\n"), dict(a=[{'a': 123, 'b': 234}], b=[{'a': 123, 'b': 2345}], x='<font color=\'black\'>{<br/>\n<font color=\'black\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": 123,</font><br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;"b": 2345,</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;"b": 234</font><br/>\n}</font><br/>\n'), dict(a=[set([1, 2, 3, 4])], b=[set([1, 2, 3, 4, 5])], x=''), dict(a=[], b=['<script>', '<script>', 1234, 1234.0, True, None, [1, 2, 3], {'a': 1}, set([1])], x='<font color=\'red\'>"&lt;script&gt;",</font><br/>\n<font color=\'red\'>"&lt;script&gt;",</font><br/>\n<font color=\'red\'>1234,</font><br/>\n<font color=\'red\'>1234.0,</font><br/>\n<font color=\'red\'>true,</font><br/>\n<font color=\'red\'>null,</font><br/>\n<font color=\'red\'>[<br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;3</font><br/>\n],</font><br/>\n<font color=\'red\'>{<br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;"a": 1</font><br/>\n}</font><br/>\n')]
    for value in values:
        result = diff_list(value['a'], value['b'], 0)
        if result != value['x']:
            print(('RE', result))
            print(('EX', value['x']))
        assert result == value['x']

def test_diff_dict():
    if False:
        return 10
    from security_monkey.common.PolicyDiff import diff_dict
    values = [dict(a={'a': 'hello', 'b': [1, 2, 3]}, b={}, x='<font color=\'green\'>"a": "hello",</font><br/>\n<font color=\'green\'>"b": [<br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'green\'>&nbsp;&nbsp;&nbsp;&nbsp;3</font><br/>\n]</font><br/>\n'), dict(a={'a': 'str'}, b={'a': 1234}, x='<font color=\'red\'>"a": 1234,</font><br/>\n<font color=\'green\'>"a": "str"</font><br/>\n'), dict(a={'a': 'str'}, b={'a': 'george'}, x='<font color=\'red\'>"a": "george",</font><br/>\n<font color=\'green\'>"a": "str"</font><br/>\n'), dict(a={}, b={'a': 'george', 'b': [1, 2, 3]}, x='<font color=\'red\'>"a": "george",</font><br/>\n<font color=\'red\'>"b": [<br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;1,</font><br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;2,</font><br/>\n<font color=\'red\'>&nbsp;&nbsp;&nbsp;&nbsp;3</font><br/>\n]</font><br/>\n')]
    for value in values:
        result = diff_dict(value['a'], value['b'], 0)
        if result != value['x']:
            print(result)
        assert result == value['x']