import pytest
from . import normalize_ret
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.core_test]

def test_requisites_onfail_any(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Call sls file containing several require_in and require.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    a:\n      cmd.run:\n        - name: exit 0\n\n    b:\n      cmd.run:\n        - name: exit 1\n\n    c:\n      cmd.run:\n        - name: exit 0\n\n    d:\n      cmd.run:\n        - name: echo itworked\n        - onfail_any:\n          - cmd: a\n          - cmd: b\n          - cmd: c\n\n    e:\n      cmd.run:\n        - name: exit 0\n\n    f:\n      cmd.run:\n        - name: exit 0\n\n    g:\n      cmd.run:\n        - name: exit 0\n\n    h:\n      cmd.run:\n        - name: echo itworked\n        - onfail_any:\n          - cmd: e\n          - cmd: f\n          - cmd: g\n    '
    expected_result = {'cmd_|-a_|-exit 0_|-run': {'__run_num__': 0, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-b_|-exit 1_|-run': {'__run_num__': 1, 'changes': True, 'comment': 'Command "exit 1" run', 'result': False}, 'cmd_|-c_|-exit 0_|-run': {'__run_num__': 2, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-d_|-echo itworked_|-run': {'__run_num__': 3, 'changes': True, 'comment': 'Command "echo itworked" run', 'result': True}, 'cmd_|-e_|-exit 0_|-run': {'__run_num__': 4, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-f_|-exit 0_|-run': {'__run_num__': 5, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-g_|-exit 0_|-run': {'__run_num__': 6, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-h_|-echo itworked_|-run': {'__run_num__': 7, 'changes': False, 'comment': 'State was not run because onfail req did not change', 'result': True}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_onfail_all(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Call sls file containing several onfail-all\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    a:\n      cmd.run:\n        - name: exit 0\n\n    b:\n      cmd.run:\n        - name: exit 0\n\n    c:\n      cmd.run:\n        - name: exit 0\n\n    d:\n      cmd.run:\n        - name: exit 1\n\n    e:\n      cmd.run:\n        - name: exit 1\n\n    f:\n      cmd.run:\n        - name: exit 1\n\n    reqs not met:\n      cmd.run:\n        - name: echo itdidntonfail\n        - onfail_all:\n          - cmd: a\n          - cmd: e\n\n    reqs also not met:\n      cmd.run:\n        - name: echo italsodidnonfail\n        - onfail_all:\n          - cmd: a\n          - cmd: b\n          - cmd: c\n\n    reqs met:\n      cmd.run:\n        - name: echo itonfailed\n        - onfail_all:\n          - cmd: d\n          - cmd: e\n          - cmd: f\n\n    reqs also met:\n      cmd.run:\n        - name: echo itonfailed\n        - onfail_all:\n          - cmd: d\n        - require:\n          - cmd: a\n    '
    expected_result = {'cmd_|-a_|-exit 0_|-run': {'__run_num__': 0, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-b_|-exit 0_|-run': {'__run_num__': 1, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-c_|-exit 0_|-run': {'__run_num__': 2, 'changes': True, 'comment': 'Command "exit 0" run', 'result': True}, 'cmd_|-d_|-exit 1_|-run': {'__run_num__': 3, 'changes': True, 'comment': 'Command "exit 1" run', 'result': False}, 'cmd_|-e_|-exit 1_|-run': {'__run_num__': 4, 'changes': True, 'comment': 'Command "exit 1" run', 'result': False}, 'cmd_|-f_|-exit 1_|-run': {'__run_num__': 5, 'changes': True, 'comment': 'Command "exit 1" run', 'result': False}, 'cmd_|-reqs also met_|-echo itonfailed_|-run': {'__run_num__': 9, 'changes': True, 'comment': 'Command "echo itonfailed" run', 'result': True}, 'cmd_|-reqs also not met_|-echo italsodidnonfail_|-run': {'__run_num__': 7, 'changes': False, 'comment': 'State was not run because onfail req did not change', 'result': True}, 'cmd_|-reqs met_|-echo itonfailed_|-run': {'__run_num__': 8, 'changes': True, 'comment': 'Command "echo itonfailed" run', 'result': True}, 'cmd_|-reqs not met_|-echo itdidntonfail_|-run': {'__run_num__': 6, 'changes': False, 'comment': 'State was not run because onfail req did not change', 'result': True}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_onfail_requisite(state, state_tree):
    if False:
        return 10
    '\n    Tests a simple state using the onfail requisite\n    '
    sls_contents = '\n    failing_state:\n      cmd.run:\n        - name: asdf\n\n    non_failing_state:\n      cmd.run:\n        - name: echo "Non-failing state"\n\n    test_failing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onfail:\n          - cmd: failing_state\n\n    test_non_failing_state:\n      cmd.run:\n        - name: echo "Should not run"\n        - onfail:\n          - cmd: non_failing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_failing_state_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'
        assert ret['cmd_|-test_non_failing_state_|-echo "Should not run"_|-run'].comment == 'State was not run because onfail req did not change'

def test_multiple_onfail_requisite(state, state_tree):
    if False:
        return 10
    '\n    test to ensure state is run even if only one\n    of the onfails fails. This is a test for the issue:\n    https://github.com/saltstack/salt/issues/22370\n    '
    sls_contents = '\n    a:\n      cmd.run:\n        - name: exit 0\n\n    b:\n      cmd.run:\n        - name: exit 1\n\n    c:\n      cmd.run:\n        - name: echo itworked\n        - onfail:\n          - cmd: a\n          - cmd: b\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-c_|-echo itworked_|-run'].changes['retcode'] == 0
        assert ret['cmd_|-c_|-echo itworked_|-run'].changes['stdout'] == 'itworked'

def test_onfail_in_requisite(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests a simple state using the onfail_in requisite\n    '
    sls_contents = '\n    failing_state:\n      cmd.run:\n        - name: asdf\n        - onfail_in:\n          - cmd: test_failing_state\n\n    non_failing_state:\n      cmd.run:\n        - name: echo "Non-failing state"\n        - onfail_in:\n          - cmd: test_non_failing_state\n\n    test_failing_state:\n      cmd.run:\n        - name: echo "Success!"\n\n    test_non_failing_state:\n      cmd.run:\n        - name: echo "Should not run"\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_failing_state_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'
        assert ret['cmd_|-test_non_failing_state_|-echo "Should not run"_|-run'].comment == 'State was not run because onfail req did not change'

def test_onfail_requisite_no_state_module(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Tests a simple state using the onfail requisite\n    '
    sls_contents = '\n    failing_state:\n      cmd.run:\n        - name: asdf\n\n    non_failing_state:\n      cmd.run:\n        - name: echo "Non-failing state"\n\n    test_failing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onfail:\n          - failing_state\n\n    test_non_failing_state:\n      cmd.run:\n        - name: echo "Should not run"\n        - onfail:\n          - non_failing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-test_failing_state_|-echo "Success!"_|-run'].comment == 'Command "echo "Success!"" run'
        assert ret['cmd_|-test_non_failing_state_|-echo "Should not run"_|-run'].comment == 'State was not run because onfail req did not change'

def test_onfail_requisite_with_duration(state, state_tree):
    if False:
        return 10
    '\n    Tests a simple state using the onfail requisite\n    '
    sls_contents = '\n    failing_state:\n      cmd.run:\n        - name: asdf\n\n    non_failing_state:\n      cmd.run:\n        - name: echo "Non-failing state"\n\n    test_failing_state:\n      cmd.run:\n        - name: echo "Success!"\n        - onfail:\n          - cmd: failing_state\n\n    test_non_failing_state:\n      cmd.run:\n        - name: echo "Should not run"\n        - onfail:\n          - cmd: non_failing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert 'duration' in ret['cmd_|-test_non_failing_state_|-echo "Should not run"_|-run']

def test_multiple_onfail_requisite_with_required(state, state_tree):
    if False:
        i = 10
        return i + 15
    '\n    test to ensure multiple states are run\n    when specified as onfails for a single state.\n    This is a test for the issue:\n    https://github.com/saltstack/salt/issues/46552\n    '
    sls_contents = '\n    a:\n      cmd.run:\n        - name: exit 1\n\n    pass:\n      cmd.run:\n        - name: exit 0\n\n    b:\n      cmd.run:\n        - name: echo b\n        - onfail:\n          - cmd: a\n\n    c:\n      cmd.run:\n        - name: echo c\n        - onfail:\n          - cmd: a\n        - require:\n          - cmd: b\n\n    d:\n      cmd.run:\n        - name: echo d\n        - onfail:\n          - cmd: a\n        - require:\n          - cmd: c\n\n    e:\n      cmd.run:\n        - name: echo e\n        - onfail:\n          - cmd: pass\n        - require:\n          - cmd: c\n\n    f:\n      cmd.run:\n        - name: echo f\n        - onfail:\n          - cmd: pass\n        - onchanges:\n          - cmd: b\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-b_|-echo b_|-run'].changes['retcode'] == 0
        assert ret['cmd_|-c_|-echo c_|-run'].changes['retcode'] == 0
        assert ret['cmd_|-d_|-echo d_|-run'].changes['retcode'] == 0
        assert ret['cmd_|-b_|-echo b_|-run'].changes['stdout'] == 'b'
        assert ret['cmd_|-c_|-echo c_|-run'].changes['stdout'] == 'c'
        assert ret['cmd_|-d_|-echo d_|-run'].changes['stdout'] == 'd'
        assert ret['cmd_|-e_|-echo e_|-run'].comment == 'State was not run because onfail req did not change'
        assert ret['cmd_|-f_|-echo f_|-run'].comment == 'State was not run because onfail req did not change'

def test_multiple_onfail_requisite_with_required_no_run(state, state_tree):
    if False:
        print('Hello World!')
    '\n    test to ensure multiple states are not run\n    when specified as onfails for a single state\n    which fails.\n    This is a test for the issue:\n    https://github.com/saltstack/salt/issues/46552\n    '
    sls_contents = '\n    a:\n      cmd.run:\n        - name: exit 0\n\n    b:\n      cmd.run:\n        - name: echo b\n        - onfail:\n          - cmd: a\n\n    c:\n      cmd.run:\n        - name: echo c\n        - onfail:\n          - cmd: a\n        - require:\n          - cmd: b\n\n    d:\n      cmd.run:\n        - name: echo d\n        - onfail:\n          - cmd: a\n        - require:\n          - cmd: c\n    '
    expected = 'State was not run because onfail req did not change'
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-b_|-echo b_|-run'].comment == expected
        assert ret['cmd_|-c_|-echo c_|-run'].comment == expected
        assert ret['cmd_|-d_|-echo d_|-run'].comment == expected