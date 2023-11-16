import pytest
from . import normalize_ret
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.core_test]

def test_requisites_onchanges_any(state, state_tree):
    if False:
        while True:
            i = 10
    '\n    Call sls file containing several require_in and require.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    another_changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - comment: non_changing_state not changed\n\n    another_non_changing_state:\n      test.succeed_without_changes:\n        - comment: another_non_changing_state not changed\n\n    # Should succeed since at least one will have changes\n    test_one_changing_states:\n      cmd.run:\n        - name: echo "Success!"\n        - onchanges_any:\n          - cmd: changing_state\n          - cmd: another_changing_state\n          - test: non_changing_state\n          - test: another_non_changing_state\n\n    test_two_non_changing_states:\n      cmd.run:\n        - name: echo "Should not run"\n        - onchanges_any:\n          - test: non_changing_state\n          - test: another_non_changing_state\n    '
    expected_result = {'cmd_|-another_changing_state_|-echo "Changed!"_|-run': {'__run_num__': 1, 'changes': True, 'comment': 'Command "echo "Changed!"" run', 'result': True}, 'cmd_|-changing_state_|-echo "Changed!"_|-run': {'__run_num__': 0, 'changes': True, 'comment': 'Command "echo "Changed!"" run', 'result': True}, 'cmd_|-test_one_changing_states_|-echo "Success!"_|-run': {'__run_num__': 4, 'changes': True, 'comment': 'Command "echo "Success!"" run', 'result': True}, 'cmd_|-test_two_non_changing_states_|-echo "Should not run"_|-run': {'__run_num__': 5, 'changes': False, 'comment': 'State was not run because none of the onchanges reqs changed', 'result': True}, 'test_|-another_non_changing_state_|-another_non_changing_state_|-succeed_without_changes': {'__run_num__': 3, 'changes': False, 'comment': 'another_non_changing_state not changed', 'result': True}, 'test_|-non_changing_state_|-non_changing_state_|-succeed_without_changes': {'__run_num__': 2, 'changes': False, 'comment': 'non_changing_state not changed', 'result': True}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_onchanges_requisite(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests a simple state using the onchanges requisite\n    '
    sls_contents = '\n    changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - comment: non_changing_state not changed\n\n    test_changing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onchanges:\n          - cmd: changing_state\n\n    test_non_changing_state:\n      cmd.run:\n        - name: echo "Should not run"\n        - onchanges:\n          - test: non_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_changing_state_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'
        assert ret['cmd_|-test_non_changing_state_|-echo "Should not run"_|-run'].comment == 'State was not run because none of the onchanges reqs changed'

def test_onchanges_requisite_multiple(state, state_tree):
    if False:
        return 10
    '\n    Tests a simple state using the onchanges requisite\n    '
    sls_contents = '\n    changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    another_changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - comment: non_changing_state not changed\n\n    another_non_changing_state:\n      test.succeed_without_changes:\n        - comment: another_non_changing_state not changed\n\n    test_two_changing_states:\n      cmd.run:\n        - name: echo "Success!"\n        - onchanges:\n          - cmd: changing_state\n          - cmd: another_changing_state\n\n    test_two_non_changing_states:\n      cmd.run:\n        - name: echo "Should not run"\n        - onchanges:\n          - test: non_changing_state\n          - test: another_non_changing_state\n\n    test_one_changing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onchanges:\n          - cmd: changing_state\n          - test: non_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_two_changing_states_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'
        assert ret['cmd_|-test_two_non_changing_states_|-echo "Should not run"_|-run'].comment == 'State was not run because none of the onchanges reqs changed'
        assert ret['cmd_|-test_one_changing_state_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'

def test_onchanges_in_requisite(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Tests a simple state using the onchanges_in requisite\n    '
    sls_contents = '\n    changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n        - onchanges_in:\n          - cmd: test_changes_expected\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - comment: non_changing_state not changed\n        - onchanges_in:\n          - cmd: test_changes_not_expected\n\n    test_changes_expected:\n      cmd.run:\n        - name: echo "Success!"\n\n    test_changes_not_expected:\n      cmd.run:\n        - name: echo "Should not run"\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_changes_expected_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'
        assert ret['cmd_|-test_changes_not_expected_|-echo "Should not run"_|-run'].comment == 'State was not run because none of the onchanges reqs changed'

def test_onchanges_requisite_no_state_module(state, state_tree):
    if False:
        while True:
            i = 10
    '\n    Tests a simple state using the onchanges requisite without state modules\n    '
    sls_contents = '\n    changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - comment: non_changing_state not changed\n\n    test_changing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onchanges:\n          - changing_state\n\n    test_non_changing_state:\n      cmd.run:\n        - name: echo "Should not run"\n        - onchanges:\n          - non_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_changing_state_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'

def test_onchanges_requisite_with_duration(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests a simple state using the onchanges requisite\n    the state will not run but results will include duration\n    '
    sls_contents = '\n    changing_state:\n      cmd.run:\n        - name: echo "Changed!"\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - comment: non_changing_state not changed\n\n    test_changing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onchanges:\n          - cmd: changing_state\n\n    test_non_changing_state:\n      cmd.run:\n        - name: echo "Should not run"\n        - onchanges:\n          - test: non_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert 'duration' in ret['cmd_|-test_non_changing_state_|-echo "Should not run"_|-run']

def test_onchanges_any_recursive_error_issues_50811(state, state_tree):
    if False:
        while True:
            i = 10
    '\n    test that onchanges_any does not causes a recursive error\n    '
    sls_contents = '\n    command-test:\n      cmd.run:\n        - name: ls\n        - onchanges_any:\n          - file: /tmp/an-unfollowed-file\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
    assert ret['command-test'].result is False