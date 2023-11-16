import pytest
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.core_test]

def test_listen_requisite(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests a simple state using the listen requisite\n    '
    sls_contents = '\n    successful_changing_state:\n      cmd.run:\n        - name: echo "Successful Change"\n\n    non_changing_state:\n      test.succeed_without_changes\n\n    test_listening_change_state:\n      cmd.run:\n        - name: echo "Listening State"\n        - listen:\n          - cmd: successful_changing_state\n\n    test_listening_non_changing_state:\n      cmd.run:\n        - name: echo "Only run once"\n        - listen:\n          - test: non_changing_state\n\n    # test that requisite resolution for listen uses ID declaration.\n    # test_listening_resolution_one and test_listening_resolution_two\n    # should both run.\n    test_listening_resolution_one:\n      cmd.run:\n        - name: echo "Successful listen resolution"\n        - listen:\n          - cmd: successful_changing_state\n\n    test_listening_resolution_two:\n      cmd.run:\n        - name: echo "Successful listen resolution"\n        - listen:\n          - cmd: successful_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        listener_state = 'cmd_|-listener_test_listening_change_state_|-echo "Listening State"_|-mod_watch'
        assert listener_state in ret
        absent_state = 'cmd_|-listener_test_listening_non_changing_state_|-echo "Only run once"_|-mod_watch'
        assert absent_state not in ret

def test_listen_in_requisite(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Tests a simple state using the listen_in requisite\n    '
    sls_contents = '\n    successful_changing_state:\n      cmd.run:\n        - name: echo "Successful Change"\n        - listen_in:\n          - cmd: test_listening_change_state\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - listen_in:\n          - cmd: test_listening_non_changing_state\n\n    test_listening_change_state:\n      cmd.run:\n        - name: echo "Listening State"\n\n    test_listening_non_changing_state:\n      cmd.run:\n        - name: echo "Only run once"\n\n    # test that requisite resolution for listen_in uses ID declaration.\n    # test_listen_in_resolution should run.\n    test_listen_in_resolution:\n      cmd.wait:\n        - name: echo "Successful listen_in resolution"\n\n    successful_changing_state_name_foo:\n      test.succeed_with_changes:\n        - name: foo\n        - listen_in:\n          - cmd: test_listen_in_resolution\n\n    successful_non_changing_state_name_foo:\n      test.succeed_without_changes:\n        - name: foo\n        - listen_in:\n          - cmd: test_listen_in_resolution\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        listener_state = 'cmd_|-listener_test_listening_change_state_|-echo "Listening State"_|-mod_watch'
        assert listener_state in ret
        absent_state = 'cmd_|-listener_test_listening_non_changing_state_|-echo "Only run once"_|-mod_watch'
        assert absent_state not in ret

def test_listen_in_requisite_resolution(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Verify listen_in requisite lookups use ID declaration to check for changes\n    '
    sls_contents = '\n    successful_changing_state:\n      cmd.run:\n        - name: echo "Successful Change"\n        - listen_in:\n          - cmd: test_listening_change_state\n\n    non_changing_state:\n      test.succeed_without_changes:\n        - listen_in:\n          - cmd: test_listening_non_changing_state\n\n    test_listening_change_state:\n      cmd.run:\n        - name: echo "Listening State"\n\n    test_listening_non_changing_state:\n      cmd.run:\n        - name: echo "Only run once"\n\n    # test that requisite resolution for listen_in uses ID declaration.\n    # test_listen_in_resolution should run.\n    test_listen_in_resolution:\n      cmd.wait:\n        - name: echo "Successful listen_in resolution"\n\n    successful_changing_state_name_foo:\n      test.succeed_with_changes:\n        - name: foo\n        - listen_in:\n          - cmd: test_listen_in_resolution\n\n    successful_non_changing_state_name_foo:\n      test.succeed_without_changes:\n        - name: foo\n        - listen_in:\n          - cmd: test_listen_in_resolution\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        listener_state = 'cmd_|-listener_test_listen_in_resolution_|-echo "Successful listen_in resolution"_|-mod_watch'
        assert listener_state in ret

def test_listen_requisite_resolution(state, state_tree):
    if False:
        return 10
    '\n    Verify listen requisite lookups use ID declaration to check for changes\n    '
    sls_contents = '\n    successful_changing_state:\n      cmd.run:\n        - name: echo "Successful Change"\n\n    non_changing_state:\n      test.succeed_without_changes\n\n    test_listening_change_state:\n      cmd.run:\n        - name: echo "Listening State"\n        - listen:\n          - cmd: successful_changing_state\n\n    test_listening_non_changing_state:\n      cmd.run:\n        - name: echo "Only run once"\n        - listen:\n          - test: non_changing_state\n\n    # test that requisite resolution for listen uses ID declaration.\n    # test_listening_resolution_one and test_listening_resolution_two\n    # should both run.\n    test_listening_resolution_one:\n      cmd.run:\n        - name: echo "Successful listen resolution"\n        - listen:\n          - cmd: successful_changing_state\n\n    test_listening_resolution_two:\n      cmd.run:\n        - name: echo "Successful listen resolution"\n        - listen:\n          - cmd: successful_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        listener_state = 'cmd_|-listener_test_listening_resolution_one_|-echo "Successful listen resolution"_|-mod_watch'
        assert listener_state in ret

def test_listen_requisite_no_state_module(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests a simple state using the listen requisite\n    '
    sls_contents = '\n    successful_changing_state:\n      cmd.run:\n        - name: echo "Successful Change"\n\n    non_changing_state:\n      test.succeed_without_changes\n\n    test_listening_change_state:\n      cmd.run:\n        - name: echo "Listening State"\n        - listen:\n          - successful_changing_state\n\n    test_listening_non_changing_state:\n      cmd.run:\n        - name: echo "Only run once"\n        - listen:\n          - non_changing_state\n\n    # test that requisite resolution for listen uses ID declaration.\n    # test_listening_resolution_one and test_listening_resolution_two\n    # should both run.\n    test_listening_resolution_one:\n      cmd.run:\n        - name: echo "Successful listen resolution"\n        - listen:\n          - successful_changing_state\n\n    test_listening_resolution_two:\n      cmd.run:\n        - name: echo "Successful listen resolution"\n        - listen:\n          - successful_changing_state\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        listener_state = 'cmd_|-listener_test_listening_change_state_|-echo "Listening State"_|-mod_watch'
        assert listener_state in ret
        absent_state = 'cmd_|-listener_test_listening_non_changing_state_|-echo "Only run once"_|-mod_watch'
        assert absent_state not in ret

def test_listen_in_requisite_resolution_names(state, state_tree):
    if False:
        return 10
    '\n    Verify listen_in requisite lookups use ID declaration to check for changes\n    and resolves magic names state variable\n    '
    sls_contents = '\n    test:\n      test.succeed_with_changes:\n        - name: test\n        - listen_in:\n          - test: service\n\n    service:\n      test.succeed_without_changes:\n        - names:\n          - nginx\n          - crond\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert 'test_|-listener_service_|-nginx_|-mod_watch' in ret
        assert 'test_|-listener_service_|-crond_|-mod_watch' in ret

def test_listen_requisite_resolution_names(state, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify listen requisite lookups use ID declaration to check for changes\n    and resolves magic names state variable\n    '
    sls_contents = '\n    test:\n      test.succeed_with_changes:\n        - name: test\n\n    service:\n      test.succeed_without_changes:\n        - names:\n          - nginx\n          - crond\n        - listen:\n          - test: test\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert 'test_|-listener_service_|-nginx_|-mod_watch' in ret
        assert 'test_|-listener_service_|-crond_|-mod_watch' in ret

def test_onlyif_req(state, subtests):
    if False:
        for i in range(10):
            print('nop')
    onlyif = [{}]
    with subtests.test(onlyif=onlyif):
        ret = state.single(name='onlyif test', fun='test.succeed_with_changes', onlyif=onlyif)
        assert ret.result is True
        assert ret.comment == 'Success!'
    onlyif = [{'fun': 'test.true'}]
    with subtests.test(onlyif=onlyif):
        ret = state.single(name='onlyif test', fun='test.succeed_without_changes', onlyif=onlyif)
        assert ret.result is True
        assert not ret.changes
        assert ret.comment == 'Success!'
    onlyif = [{'fun': 'test.false'}]
    with subtests.test(onlyif=onlyif):
        ret = state.single(name='onlyif test', fun='test.fail_with_changes', onlyif=onlyif)
        assert ret.result is True
        assert not ret.changes
        assert ret.comment == 'onlyif condition is false'
    onlyif = [{'fun': 'test.true'}]
    with subtests.test(onlyif=onlyif):
        ret = state.single(name='onlyif test', fun='test.fail_with_changes', onlyif=onlyif)
        assert ret.result is False
        assert ret.changes
        assert ret.comment == 'Failure!'

def test_listen_requisite_not_exist(state, state_tree):
    if False:
        print('Hello World!')
    '\n    Tests a simple state using the listen requisite\n    when the state id does not exist\n    '
    sls_contents = '\n    successful_changing_state:\n      cmd.run:\n        - name: echo "Successful Change"\n\n    non_changing_state:\n      test.succeed_without_changes\n\n    test_listening_change_state:\n      cmd.run:\n        - name: echo "Listening State"\n        - listen:\n          - cmd: successful_changing_state\n\n    test_listening_non_changing_state:\n      cmd.run:\n        - name: echo "Only run once"\n        - listen:\n          - test: non_changing_state_not_exist\n    '
    with pytest.helpers.temp_file('requisite.sls', sls_contents, state_tree):
        ret = state.sls('requisite')
        assert ret.raw['Listen_Error_|-listen_non_changing_state_not_exist_|-listen_test_|-Listen_Error']['comment'] == 'Referenced state test: non_changing_state_not_exist does not exist'