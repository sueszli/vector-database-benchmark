"""
    :codeauthor: Nicole Thomas <nicole@saltstack.com>
"""
import logging
import pytest
import salt.exceptions
import salt.state
import salt.utils.files
import salt.utils.platform
from tests.support.mock import MagicMock, patch
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.core_test]

@pytest.fixture
def state_obj(minion_opts):
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.state.State._gather_pillar'):
        yield salt.state.State(minion_opts)

def test_format_slots_no_slots(state_obj):
    if False:
        return 10
    '\n    Test the format slots keeps data without slots untouched.\n    '
    cdata = {'args': ['arg'], 'kwargs': {'key': 'val'}}
    state_obj.format_slots(cdata)
    assert cdata == {'args': ['arg'], 'kwargs': {'key': 'val'}}

def test_format_slots_arg(state_obj):
    if False:
        while True:
            i = 10
    '\n    Test the format slots is calling a slot specified in args with corresponding arguments.\n    '
    cdata = {'args': ['__slot__:salt:mod.fun(fun_arg, fun_key=fun_val)'], 'kwargs': {'key': 'val'}}
    mock = MagicMock(return_value='fun_return')
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': ['fun_return'], 'kwargs': {'key': 'val'}}

def test_format_slots_dict_arg(state_obj):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the format slots is calling a slot specified in dict arg.\n    '
    cdata = {'args': [{'subarg': '__slot__:salt:mod.fun(fun_arg, fun_key=fun_val)'}], 'kwargs': {'key': 'val'}}
    mock = MagicMock(return_value='fun_return')
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': [{'subarg': 'fun_return'}], 'kwargs': {'key': 'val'}}

def test_format_slots_listdict_arg(state_obj):
    if False:
        while True:
            i = 10
    '\n    Test the format slots is calling a slot specified in list containing a dict.\n    '
    cdata = {'args': [[{'subarg': '__slot__:salt:mod.fun(fun_arg, fun_key=fun_val)'}]], 'kwargs': {'key': 'val'}}
    mock = MagicMock(return_value='fun_return')
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': [[{'subarg': 'fun_return'}]], 'kwargs': {'key': 'val'}}

def test_format_slots_liststr_arg(state_obj):
    if False:
        return 10
    '\n    Test the format slots is calling a slot specified in list containing a dict.\n    '
    cdata = {'args': [['__slot__:salt:mod.fun(fun_arg, fun_key=fun_val)']], 'kwargs': {'key': 'val'}}
    mock = MagicMock(return_value='fun_return')
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': [['fun_return']], 'kwargs': {'key': 'val'}}

def test_format_slots_kwarg(state_obj):
    if False:
        print('Hello World!')
    '\n    Test the format slots is calling a slot specified in kwargs with corresponding arguments.\n    '
    cdata = {'args': ['arg'], 'kwargs': {'key': '__slot__:salt:mod.fun(fun_arg, fun_key=fun_val)'}}
    mock = MagicMock(return_value='fun_return')
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': ['arg'], 'kwargs': {'key': 'fun_return'}}

def test_format_slots_multi(state_obj):
    if False:
        i = 10
        return i + 15
    '\n    Test the format slots is calling all slots with corresponding arguments when multiple slots\n    specified.\n    '
    cdata = {'args': ['__slot__:salt:test_mod.fun_a(a_arg, a_key=a_kwarg)', '__slot__:salt:test_mod.fun_b(b_arg, b_key=b_kwarg)'], 'kwargs': {'kw_key_1': '__slot__:salt:test_mod.fun_c(c_arg, c_key=c_kwarg)', 'kw_key_2': '__slot__:salt:test_mod.fun_d(d_arg, d_key=d_kwarg)'}}
    mock_a = MagicMock(return_value='fun_a_return')
    mock_b = MagicMock(return_value='fun_b_return')
    mock_c = MagicMock(return_value='fun_c_return')
    mock_d = MagicMock(return_value='fun_d_return')
    with patch.dict(state_obj.functions, {'test_mod.fun_a': mock_a, 'test_mod.fun_b': mock_b, 'test_mod.fun_c': mock_c, 'test_mod.fun_d': mock_d}):
        state_obj.format_slots(cdata)
    mock_a.assert_called_once_with('a_arg', a_key='a_kwarg')
    mock_b.assert_called_once_with('b_arg', b_key='b_kwarg')
    mock_c.assert_called_once_with('c_arg', c_key='c_kwarg')
    mock_d.assert_called_once_with('d_arg', d_key='d_kwarg')
    assert cdata == {'args': ['fun_a_return', 'fun_b_return'], 'kwargs': {'kw_key_1': 'fun_c_return', 'kw_key_2': 'fun_d_return'}}

def test_format_slots_malformed(state_obj):
    if False:
        return 10
    '\n    Test the format slots keeps malformed slots untouched.\n    '
    sls_data = {'args': ['__slot__:NOT_SUPPORTED:not.called()', '__slot__:salt:not.called(', '__slot__:salt:', '__slot__:salt', '__slot__:', '__slot__'], 'kwargs': {'key3': '__slot__:NOT_SUPPORTED:not.called()', 'key4': '__slot__:salt:not.called(', 'key5': '__slot__:salt:', 'key6': '__slot__:salt', 'key7': '__slot__:', 'key8': '__slot__'}}
    cdata = sls_data.copy()
    mock = MagicMock(return_value='return')
    with patch.dict(state_obj.functions, {'not.called': mock}):
        state_obj.format_slots(cdata)
    mock.assert_not_called()
    assert cdata == sls_data

def test_slot_traverse_dict(state_obj):
    if False:
        return 10
    '\n    Test the slot parsing of dict response.\n    '
    cdata = {'args': ['arg'], 'kwargs': {'key': '__slot__:salt:mod.fun(fun_arg, fun_key=fun_val).key1'}}
    return_data = {'key1': 'value1'}
    mock = MagicMock(return_value=return_data)
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': ['arg'], 'kwargs': {'key': 'value1'}}

def test_slot_append(state_obj):
    if False:
        i = 10
        return i + 15
    '\n    Test the slot parsing of dict response.\n    '
    cdata = {'args': ['arg'], 'kwargs': {'key': '__slot__:salt:mod.fun(fun_arg, fun_key=fun_val).key1 ~ thing~'}}
    return_data = {'key1': 'value1'}
    mock = MagicMock(return_value=return_data)
    with patch.dict(state_obj.functions, {'mod.fun': mock}):
        state_obj.format_slots(cdata)
    mock.assert_called_once_with('fun_arg', fun_key='fun_val')
    assert cdata == {'args': ['arg'], 'kwargs': {'key': 'value1thing~'}}

@pytest.mark.skip_on_spawning_platform(reason='Skipped until parallel states can be fixed on spawning platforms.')
def test_format_slots_parallel(state_obj):
    if False:
        while True:
            i = 10
    '\n    Test if slots work with "parallel: true".\n    '
    high_data = {'always-changes-and-succeeds': {'test': [{'changes': True}, {'comment': '__slot__:salt:test.echo(fun_return)'}, {'parallel': True}, 'configurable_test_state', {'order': 10000}], '__env__': 'base', '__sls__': 'parallel_slots'}}
    state_obj.jid = '123'
    res = state_obj.call_high(high_data)
    state_obj.jid = None
    [(_, data)] = res.items()
    assert data['comment'] == 'fun_return'