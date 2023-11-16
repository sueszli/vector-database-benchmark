"""
:codeauthor: Shane Lee <slee@saltstack.com>
"""
import copy
import pytest
import salt.config
import salt.loader
import salt.states.win_lgpo as win_lgpo
import salt.utils.platform
import salt.utils.stringutils
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules(minion_opts):
    if False:
        while True:
            i = 10
    utils = salt.loader.utils(minion_opts)
    modules = salt.loader.minion_mods(minion_opts, utils=utils)
    return {win_lgpo: {'__opts__': copy.deepcopy(minion_opts), '__salt__': modules, '__utils__': utils}}

@pytest.fixture
def policy_clear():
    if False:
        while True:
            i = 10
    try:
        computer_policy = {'Point and Print Restrictions': 'Not Configured'}
        with patch.dict(win_lgpo.__opts__, {'test': False}):
            win_lgpo.set_(name='test_state', computer_policy=computer_policy)
        yield
    finally:
        computer_policy = {'Point and Print Restrictions': 'Not Configured'}
        with patch.dict(win_lgpo.__opts__, {'test': False}):
            win_lgpo.set_(name='test_state', computer_policy=computer_policy)

@pytest.fixture
def policy_set():
    if False:
        i = 10
        return i + 15
    try:
        computer_policy = {'Point and Print Restrictions': {'Users can only point and print to these servers': True, 'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'Users can only point and print to machines in their forest': True, 'When installing drivers for a new connection': 'Show warning and elevation prompt', 'When updating drivers for an existing connection': 'Show warning only'}}
        with patch.dict(win_lgpo.__opts__, {'test': False}):
            win_lgpo.set_(name='test_state', computer_policy=computer_policy)
        yield
    finally:
        computer_policy = {'Point and Print Restrictions': 'Not Configured'}
        with patch.dict(win_lgpo.__opts__, {'test': False}):
            win_lgpo.set_(name='test_state', computer_policy=computer_policy)

def test__compare_policies_string():
    if False:
        return 10
    '\n    ``_compare_policies`` should only return ``True`` when the string values\n    are the same. All other scenarios should return ``False``\n    '
    compare_string = 'Salty test'
    assert win_lgpo._compare_policies(compare_string, compare_string)
    assert not win_lgpo._compare_policies(compare_string, 'Not the same')
    assert not win_lgpo._compare_policies(compare_string, ['item1', 'item2'])
    assert not win_lgpo._compare_policies(compare_string, {'key': 'value'})
    assert not win_lgpo._compare_policies(compare_string, None)

def test__compare_policies_list():
    if False:
        while True:
            i = 10
    '\n    ``_compare_policies`` should only return ``True`` when the lists are the\n    same. All other scenarios should return ``False``\n    '
    compare_list = ['Salty', 'test']
    assert win_lgpo._compare_policies(compare_list, compare_list)
    assert not win_lgpo._compare_policies(compare_list, ['Not', 'the', 'same'])
    assert not win_lgpo._compare_policies(compare_list, 'Not a list')
    assert not win_lgpo._compare_policies(compare_list, {'key': 'value'})
    assert not win_lgpo._compare_policies(compare_list, None)

def test__compare_policies_dict():
    if False:
        print('Hello World!')
    '\n    ``_compare_policies`` should only return ``True`` when the dicts are the\n    same. All other scenarios should return ``False``\n    '
    compare_dict = {'Salty': 'test'}
    assert win_lgpo._compare_policies(compare_dict, compare_dict)
    assert not win_lgpo._compare_policies(compare_dict, {'key': 'value'})
    assert not win_lgpo._compare_policies(compare_dict, 'Not a dict')
    assert not win_lgpo._compare_policies(compare_dict, ['Not', 'a', 'dict'])
    assert not win_lgpo._compare_policies(compare_dict, None)

def test__compare_policies_integer():
    if False:
        i = 10
        return i + 15
    '\n    ``_compare_policies`` should only return ``True`` when the integer\n    values are the same. All other scenarios should return ``False``\n    '
    compare_integer = 1
    assert win_lgpo._compare_policies(compare_integer, compare_integer)
    assert not win_lgpo._compare_policies(compare_integer, 0)
    assert not win_lgpo._compare_policies(compare_integer, ['item1', 'item2'])
    assert not win_lgpo._compare_policies(compare_integer, {'key': 'value'})
    assert not win_lgpo._compare_policies(compare_integer, None)

@pytest.mark.skip_unless_on_windows
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_current_element_naming_style(policy_clear):
    if False:
        print('Hello World!')
    '\n    Ensure that current naming style works properly.\n    '
    computer_policy = {'Point and Print Restrictions': {'Users can only point and print to these servers': True, 'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'Users can only point and print to machines in their forest': True, 'When installing drivers for a new connection': 'Show warning and elevation prompt', 'When updating drivers for an existing connection': 'Show warning only'}}
    with patch.dict(win_lgpo.__opts__, {'test': False}):
        result = win_lgpo.set_(name='test_state', computer_policy=computer_policy)
        result = win_lgpo._convert_to_unicode(result)
    expected = {'Point and Print Restrictions': {'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'When installing drivers for a new connection': 'Show warning and elevation prompt', 'Users can only point and print to machines in their forest': True, 'Users can only point and print to these servers': True, 'When updating drivers for an existing connection': 'Show warning only'}}
    assert result['changes']['new']['Computer Configuration'] == expected

@pytest.mark.skip_unless_on_windows
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_old_element_naming_style(policy_clear):
    if False:
        while True:
            i = 10
    '\n    Ensure that the old naming style is converted to new and a warning is\n    returned\n    '
    computer_policy = {'Point and Print Restrictions': {'Users can only point and print to these servers': True, 'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'Users can only point and print to machines in their forest': True, 'Security Prompts: When installing drivers for a new connection': 'Show warning and elevation prompt', 'When updating drivers for an existing connection': 'Show warning only'}}
    with patch.dict(win_lgpo.__opts__, {'test': False}):
        result = win_lgpo.set_(name='test_state', computer_policy=computer_policy)
    assert result['changes'] == {}
    expected = 'The LGPO module changed the way it gets policy element names.\n"Security Prompts: When installing drivers for a new connection" is no longer valid.\nPlease use "When installing drivers for a new connection" instead.'
    assert result['comment'] == expected

@pytest.mark.skip_unless_on_windows
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_invalid_elements():
    if False:
        return 10
    computer_policy = {'Point and Print Restrictions': {'Invalid element spongebob': True, 'Invalid element squidward': False}}
    with patch.dict(win_lgpo.__opts__, {'test': False}):
        result = win_lgpo.set_(name='test_state', computer_policy=computer_policy)
    expected = {'changes': {}, 'comment': 'Invalid element name: Invalid element squidward\nInvalid element name: Invalid element spongebob', 'name': 'test_state', 'result': False}
    assert result['changes'] == expected['changes']
    assert 'Invalid element squidward' in result['comment']
    assert 'Invalid element spongebob' in result['comment']
    assert not expected['result']

@pytest.mark.skip_unless_on_windows
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_current_element_naming_style_true(policy_set):
    if False:
        return 10
    '\n    Test current naming style with test=True\n    '
    computer_policy = {'Point and Print Restrictions': {'Users can only point and print to these servers': True, 'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'Users can only point and print to machines in their forest': True, 'When installing drivers for a new connection': 'Show warning and elevation prompt', 'When updating drivers for an existing connection': 'Show warning only'}}
    with patch.dict(win_lgpo.__opts__, {'test': True}):
        result = win_lgpo.set_(name='test_state', computer_policy=computer_policy)
    expected = {'changes': {}, 'comment': 'All specified policies are properly configured'}
    assert result['changes'] == expected['changes']
    assert result['result']
    assert result['comment'] == expected['comment']

@pytest.mark.skip_unless_on_windows
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_old_element_naming_style_true(policy_set):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test old naming style with test=True. Should not make changes but return a\n    warning\n    '
    computer_policy = {'Point and Print Restrictions': {'Users can only point and print to these servers': True, 'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'Users can only point and print to machines in their forest': True, 'Security Prompts: When installing drivers for a new connection': 'Show warning and elevation prompt', 'When updating drivers for an existing connection': 'Show warning only'}}
    with patch.dict(win_lgpo.__opts__, {'test': True}):
        result = win_lgpo.set_(name='test_state', computer_policy=computer_policy)
    expected = {'changes': {}, 'comment': 'The LGPO module changed the way it gets policy element names.\n"Security Prompts: When installing drivers for a new connection" is no longer valid.\nPlease use "When installing drivers for a new connection" instead.'}
    assert result['changes'] == expected['changes']
    assert not result['result']
    assert result['comment'] == expected['comment']

@pytest.mark.skip_unless_on_windows
@pytest.mark.destructive_test
@pytest.mark.slow_test
def test_invalid_elements_true():
    if False:
        for i in range(10):
            print('nop')
    computer_policy = {'Point and Print Restrictions': {'Invalid element spongebob': True, 'Invalid element squidward': False}}
    with patch.dict(win_lgpo.__opts__, {'test': True}):
        result = win_lgpo.set_(name='test_state', computer_policy=computer_policy)
    expected = {'changes': {}, 'comment': 'Invalid element name: Invalid element squidward\nInvalid element name: Invalid element spongebob', 'name': 'test_state', 'result': False}
    assert result['changes'] == expected['changes']
    assert 'Invalid element squidward' in result['comment']
    assert 'Invalid element spongebob' in result['comment']
    assert not expected['result']