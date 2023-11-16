import pytest
from . import normalize_ret
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.core_test]

def test_requisites_full_sls_prereq(state, state_tree):
    if False:
        i = 10
        return i + 15
    '\n    Test the sls special command in requisites\n    '
    full_sls_contents = '\n    B:\n      cmd.run:\n        - name: echo B\n    C:\n      cmd.run:\n        - name: echo C\n    '
    sls_contents = '\n    include:\n      - fullsls\n    A:\n      cmd.run:\n        - name: echo A\n        - prereq:\n          - sls: fullsls\n    '
    expected_result = {'cmd_|-A_|-echo A_|-run': {'__run_num__': 0, 'comment': 'Command "echo A" run', 'result': True, 'changes': True}, 'cmd_|-B_|-echo B_|-run': {'__run_num__': 1, 'comment': 'Command "echo B" run', 'result': True, 'changes': True}, 'cmd_|-C_|-echo C_|-run': {'__run_num__': 2, 'comment': 'Command "echo C" run', 'result': True, 'changes': True}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree), pytest.helpers.temp_file('fullsls.sls', full_sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_prereq_simple_ordering_and_errors_1(state, state_tree):
    if False:
        i = 10
        return i + 15
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # B --+\n    #     |\n    # C <-+ ----+\n    #           |\n    # A <-------+\n\n    # runs after C\n    A:\n      cmd.run:\n        - name: echo A third\n        # is running in test mode before C\n        # C gets executed first if this states modify something\n        - prereq_in:\n          - cmd: C\n\n    # runs before C\n    B:\n      cmd.run:\n        - name: echo B first\n        # will test C and be applied only if C changes,\n        # and then will run before C\n        - prereq:\n          - cmd: C\n    C:\n      cmd.run:\n        - name: echo C second\n\n    # will fail with "The following requisites were not found"\n    I:\n      cmd.run:\n        - name: echo I\n        - prereq:\n          - cmd: Z\n    J:\n      cmd.run:\n        - name: echo J\n        - prereq:\n          - foobar: A\n    '
    expected_result = {'cmd_|-A_|-echo A third_|-run': {'__run_num__': 2, 'comment': 'Command "echo A third" run', 'result': True, 'changes': True}, 'cmd_|-B_|-echo B first_|-run': {'__run_num__': 0, 'comment': 'Command "echo B first" run', 'result': True, 'changes': True}, 'cmd_|-C_|-echo C second_|-run': {'__run_num__': 1, 'comment': 'Command "echo C second" run', 'result': True, 'changes': True}, 'cmd_|-I_|-echo I_|-run': {'__run_num__': 3, 'comment': 'The following requisites were not found:\n' + '                   prereq:\n' + '                       cmd: Z\n', 'result': False, 'changes': False}, 'cmd_|-J_|-echo J_|-run': {'__run_num__': 4, 'comment': 'The following requisites were not found:\n' + '                   prereq:\n' + '                       foobar: A\n', 'result': False, 'changes': False}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_prereq_simple_ordering_and_errors_2(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # B --+\n    #     |\n    # C <-+ ----+\n    #           |\n    # A <-------+\n\n    # runs after C\n    A:\n      cmd.run:\n        - name: echo A third\n        # is running in test mode before C\n        # C gets executed first if this states modify something\n        - prereq_in:\n           cmd: C\n\n    # runs before C\n    B:\n      cmd.run:\n        - name: echo B first\n        # will test C and be applied only if C changes,\n        # and then will run before C\n        - prereq:\n            cmd: C\n    C:\n      cmd.run:\n        - name: echo C second\n    '
    errmsg = "The prereq statement in state 'B' in SLS 'requisite' needs to be formed as a list"
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret.failed
        assert ret.errors == [errmsg]

def test_requisites_prereq_simple_ordering_and_errors_3(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # B --+\n    #     |\n    # C <-+ ----+\n    #           |\n    # A <-------+\n\n    # runs after C\n    A:\n      cmd.run:\n        - name: echo A third\n        # is running in test mode before C\n        # C gets executed first if this states modify something\n        - prereq_in:\n          - C\n\n    # runs before C\n    B:\n      cmd.run:\n        - name: echo B first\n        # will test C and be applied only if C changes,\n        # and then will run before C\n        - prereq:\n          - C\n    C:\n      cmd.run:\n        - name: echo C second\n\n    # will fail with "The following requisites were not found"\n    I:\n      cmd.run:\n        - name: echo I\n        - prereq:\n          - Z\n        '
    expected_result = {'cmd_|-A_|-echo A third_|-run': {'__run_num__': 2, 'comment': 'Command "echo A third" run', 'result': True, 'changes': True}, 'cmd_|-B_|-echo B first_|-run': {'__run_num__': 0, 'comment': 'Command "echo B first" run', 'result': True, 'changes': True}, 'cmd_|-C_|-echo C second_|-run': {'__run_num__': 1, 'comment': 'Command "echo C second" run', 'result': True, 'changes': True}, 'cmd_|-I_|-echo I_|-run': {'__run_num__': 3, 'comment': 'The following requisites were not found:\n' + '                   prereq:\n' + '                       id: Z\n', 'result': False, 'changes': False}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_prereq_simple_ordering_and_errors_4(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # Theory:\n    #\n    # C <--+ <--+ <-+ <-+\n    #      |    |   |   |\n    # A ---+    |   |   |\n    #           |   |   |\n    # B --------+   |   |\n    #               |   |\n    # D-------------+   |\n    #                   |\n    # E-----------------+\n\n    # runs after C\n    A:\n      cmd.run:\n        - name: echo A\n        # is running in test mode before C\n        # C gets executed first if this states modify something\n        - prereq_in:\n          - cmd: C\n\n    B:\n      cmd.run:\n        - name: echo B\n\n    # runs before D and B\n    C:\n      cmd.run:\n        - name: echo C\n        # will test D and be applied only if D changes,\n        # and then will run before D. Same for B\n        - prereq:\n          - cmd: B\n          - cmd: D\n\n    D:\n      cmd.run:\n        - name: echo D\n\n    E:\n      cmd.run:\n        - name: echo E\n        # is running in test mode before C\n        # C gets executed first if this states modify something\n        - prereq_in:\n          - cmd: C\n    '
    expected_result = {'cmd_|-A_|-echo A_|-run': {'__run_num__': 1, 'comment': 'Command "echo A" run', 'result': True, 'changes': True}, 'cmd_|-B_|-echo B_|-run': {'__run_num__': 2, 'comment': 'Command "echo B" run', 'result': True, 'changes': True}, 'cmd_|-C_|-echo C_|-run': {'__run_num__': 0, 'comment': 'Command "echo C" run', 'result': True, 'changes': True}, 'cmd_|-D_|-echo D_|-run': {'__run_num__': 3, 'comment': 'Command "echo D" run', 'result': True, 'changes': True}, 'cmd_|-E_|-echo E_|-run': {'__run_num__': 4, 'comment': 'Command "echo E" run', 'result': True, 'changes': True}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_prereq_simple_ordering_and_errors_5(state, state_tree):
    if False:
        i = 10
        return i + 15
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # A --+\n    #     |\n    # B <-+ ----+\n    #           |\n    # C <-------+\n\n    # runs before A and/or B\n    A:\n      cmd.run:\n        - name: echo A first\n        # is running in test mode before B/C\n        - prereq:\n          - cmd: B\n          - cmd: C\n\n    # always has to run\n    B:\n      cmd.run:\n        - name: echo B second\n\n    # never has to run\n    C:\n      cmd.wait:\n        - name: echo C third\n    '
    expected_result = {'cmd_|-A_|-echo A first_|-run': {'__run_num__': 0, 'comment': 'Command "echo A first" run', 'result': True, 'changes': True}, 'cmd_|-B_|-echo B second_|-run': {'__run_num__': 1, 'comment': 'Command "echo B second" run', 'result': True, 'changes': True}, 'cmd_|-C_|-echo C third_|-wait': {'__run_num__': 2, 'comment': '', 'result': True, 'changes': False}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_prereq_simple_ordering_and_errors_6(state, state_tree):
    if False:
        i = 10
        return i + 15
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # issue #8211\n    #             expected rank\n    # B --+             1\n    #     |\n    # C <-+ ----+       2/3\n    #           |\n    # D ---+    |       3/2\n    #      |    |\n    # A <--+ <--+       4\n    #\n    #             resulting rank\n    # D --+\n    #     |\n    # A <-+ <==+\n    #          |\n    # B --+    +--> unrespected A prereq_in C (FAILURE)\n    #     |    |\n    # C <-+ ===+\n\n    # runs after C\n    A:\n      cmd.run:\n        - name: echo A fourth\n        # is running in test mode before C\n        # C gets executed first if this states modify something\n        - prereq_in:\n          - cmd: C\n\n    # runs before C\n    B:\n      cmd.run:\n        - name: echo B first\n        # will test C and be applied only if C changes,\n        # and then will run before C\n        - prereq:\n          - cmd: C\n\n    C:\n      cmd.run:\n        - name: echo C second\n        # replacing A prereq_in C by theses lines\n        # changes nothing actually\n        #- prereq:\n        #  - cmd: A\n\n    # Removing D, A gets executed after C\n    # as described in (A prereq_in C)\n    # runs before A\n    D:\n      cmd.run:\n        - name: echo D third\n        # will test A and be applied only if A changes,\n        # and then will run before A\n        - prereq:\n          - cmd: A\n    '
    expected_result = {'cmd_|-A_|-echo A fourth_|-run': {'__run_num__': 3, 'comment': 'Command "echo A fourth" run', 'result': True, 'changes': True}, 'cmd_|-B_|-echo B first_|-run': {'__run_num__': 0, 'comment': 'Command "echo B first" run', 'result': True, 'changes': True}, 'cmd_|-C_|-echo C second_|-run': {'__run_num__': 1, 'comment': 'Command "echo C second" run', 'result': True, 'changes': True}, 'cmd_|-D_|-echo D third_|-run': {'__run_num__': 2, 'comment': 'Command "echo D third" run', 'result': True, 'changes': True}}
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        result = normalize_ret(ret.raw)
        assert result == expected_result

def test_requisites_prereq_simple_ordering_and_errors_7(state, state_tree):
    if False:
        while True:
            i = 10
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    # will fail with \'Cannot extend ID Z (...) not part of the high state.\'\n    # and not "The following requisites were not found" like in yaml list syntax\n    I:\n      cmd.run:\n        - name: echo I\n        - prereq:\n          - cmd: Z\n    '
    errmsg = 'The following requisites were not found:\n                   prereq:\n                       cmd: Z\n'
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-I_|-echo I_|-run'].comment == errmsg

def test_requisites_prereq_simple_ordering_and_errors_8(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    A:\n      cmd.run:\n        - name: echo A\n\n    B:\n      cmd.run:\n        - name: echo B\n        - prereq:\n          - foobar: A\n    '
    errmsg = 'The following requisites were not found:\n                   prereq:\n                       foobar: A\n'
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-B_|-echo B_|-run'].comment == errmsg

def test_requisites_prereq_simple_ordering_and_errors_9(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    A:\n      cmd.run:\n        - name: echo A\n\n    B:\n      cmd.run:\n        - name: echo B\n        - prereq:\n          - foobar: C\n    '
    errmsg = 'The following requisites were not found:\n                   prereq:\n                       foobar: C\n'
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['cmd_|-B_|-echo B_|-run'].comment == errmsg

@pytest.mark.skip('issue #8210 : prereq recursion undetected')
def test_requisites_prereq_simple_ordering_and_errors_10(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Call sls file containing several prereq_in and prereq.\n\n    Ensure that some of them are failing and that the order is right.\n    '
    sls_contents = '\n    A:\n      cmd.run:\n        - name: echo A\n        - prereq_in:\n          - cmd: B\n    B:\n      cmd.run:\n        - name: echo B\n        - prereq_in:\n          - cmd: A\n    '
    errmsg = 'A recursive requisite was found, SLS "requisites.prereq_recursion_error" ID "B" ID "A"'
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret.failed
        assert ret.errors == [errmsg]

def test_infinite_recursion_sls_prereq(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    sls_contents = '\n    include:\n      - requisite2\n    A:\n      test.succeed_without_changes:\n        - name: A\n        - prereq:\n          - sls: requisite2\n    '
    sls_2_contents = '\n    B:\n      test.succeed_without_changes:\n        - name: B\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree), pytest.helpers.temp_file('requisite2.sls', sls_2_contents, state_tree):
        ret = state.sls('requisite')
        for state_return in ret:
            assert state_return.result is True

def test_infinite_recursion_prereq(state, state_tree):
    if False:
        while True:
            i = 10
    sls_contents = '\n    A:\n      test.nop:\n        - prereq:\n          - test: B\n    B:\n      test.nop:\n        - require:\n          - name: non-existant\n    C:\n      test.nop:\n        - require:\n          - test: B\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        for state_return in ret:
            assert state_return.result is False

def test_infinite_recursion_prereq2(state, state_tree):
    if False:
        print('Hello World!')
    sls_contents = '\n    A:\n      test.nop:\n        - prereq:\n          - test: B\n    B:\n      test.nop:\n        - require:\n          - test: D\n    C:\n      test.nop:\n        - require:\n          - test: B\n    D:\n      test.nop: []\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        for state_return in ret:
            assert state_return.result is True

def test_requisites_prereq_fail_in_prereq(state, state_tree):
    if False:
        return 10
    sls_contents = '\n    State A:\n      test.configurable_test_state:\n        - result: True\n        - changes: True\n        - name: fail\n\n    State B:\n      test.configurable_test_state:\n        - changes: True\n        - result: False\n        - prereq:\n          - test: State A\n\n    State C:\n      test.nop:\n        - onchanges:\n          - test: State A\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret['test_|-State A_|-fail_|-configurable_test_state'].result is None
        assert ret['test_|-State A_|-fail_|-configurable_test_state'].full_return['changes'] == {}
        assert not ret['test_|-State B_|-State B_|-configurable_test_state'].result
        assert ret['test_|-State C_|-State C_|-nop'].result
        assert not ret['test_|-State C_|-State C_|-nop'].full_return['__state_ran__']
        assert ret['test_|-State C_|-State C_|-nop'].full_return['comment'] == 'State was not run because none of the onchanges reqs changed'