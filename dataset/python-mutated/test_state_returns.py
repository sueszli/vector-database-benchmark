"""
    :codeauthor: Nicole Thomas <nicole@saltstack.com>
"""
import logging
import pytest
from salt.utils.decorators import state as statedecorators
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.core_test]

def test_state_output_check_changes_is_dict():
    if False:
        print('Hello World!')
    '\n    Test that changes key contains a dictionary.\n    :return:\n    '
    data = {'changes': []}
    out = statedecorators.OutputUnifier('content_check')(lambda : data)()
    assert "'Changes' should be a dictionary" in out['comment']
    assert not out['result']

def test_state_output_check_return_is_dict():
    if False:
        return 10
    '\n    Test for the entire return is a dictionary\n    :return:\n    '
    data = ['whatever']
    out = statedecorators.OutputUnifier('content_check')(lambda : data)()
    assert 'Malformed state return. Data must be a dictionary type' in out['comment']
    assert not out['result']

def test_state_output_check_return_has_nrc():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for name/result/comment keys are inside the return.\n    :return:\n    '
    data = {'arbitrary': 'data', 'changes': {}}
    out = statedecorators.OutputUnifier('content_check')(lambda : data)()
    assert ' The following keys were not present in the state return: name, result, comment' in out['comment']
    assert not out['result']

def test_state_output_unifier_comment_is_not_list():
    if False:
        print('Hello World!')
    '\n    Test for output is unified so the comment is converted to a multi-line string\n    :return:\n    '
    data = {'comment': ['data', 'in', 'the', 'list'], 'changes': {}, 'name': None, 'result': 'fantastic!'}
    expected = {'comment': 'data\nin\nthe\nlist', 'changes': {}, 'name': None, 'result': True}
    assert statedecorators.OutputUnifier('unify')(lambda : data)() == expected
    data = {'comment': ['data', 'in', 'the', 'list'], 'changes': {}, 'name': None, 'result': None}
    expected = 'data\nin\nthe\nlist'
    assert statedecorators.OutputUnifier('unify')(lambda : data)()['comment'] == expected

def test_state_output_unifier_result_converted_to_true():
    if False:
        print('Hello World!')
    '\n    Test for output is unified so the result is converted to True\n    :return:\n    '
    data = {'comment': ['data', 'in', 'the', 'list'], 'changes': {}, 'name': None, 'result': 'Fantastic'}
    assert statedecorators.OutputUnifier('unify')(lambda : data)()['result'] is True

def test_state_output_unifier_result_converted_to_false():
    if False:
        return 10
    '\n    Test for output is unified so the result is converted to False\n    :return:\n    '
    data = {'comment': ['data', 'in', 'the', 'list'], 'changes': {}, 'name': None, 'result': ''}
    assert statedecorators.OutputUnifier('unify')(lambda : data)()['result'] is False