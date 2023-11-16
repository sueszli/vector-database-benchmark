import pytest
pytestmark = [pytest.mark.slow_test]

def test_orchestrate_output(salt_run_cli, salt_minion, salt_master):
    if False:
        print('Hello World!')
    '\n    Ensure the orchestrate runner outputs useful state data.\n\n    In Issue #31330, the output only contains [\'outputter:\', \'    highstate\'],\n    and not the full stateful return. This tests ensures we don\'t regress in that\n    manner again.\n\n    Also test against some sample "good" output that would be included in a correct\n    orchestrate run.\n    '
    bad_out = ['outputter:', '    highstate']
    good_out = ['    Function: salt.state', '      Result: True', 'Succeeded: 1 (changed=1)', 'Failed:    0', 'Total states run:     1']
    sls_contents = '\n    call_sleep_state:\n      salt.state:\n        - tgt: {}\n        - sls: simple-ping\n    '.format(salt_minion.id)
    simple_ping_sls = '\n    simple-ping:\n      module.run:\n        - name: test.ping\n    '
    with salt_master.state_tree.base.temp_file('orch-test.sls', sls_contents), salt_master.state_tree.base.temp_file('simple-ping.sls', simple_ping_sls):
        ret = salt_run_cli.run('--out=highstate', 'state.orchestrate', 'orch-test')
        assert ret.returncode == 0
        ret_output = ret.stdout.splitlines()
        assert bad_out != ret_output
        assert len(ret_output) > 2
        for item in good_out:
            assert item in ret_output

def test_orchestrate_state_output_with_salt_function(salt_run_cli, salt_minion, salt_master):
    if False:
        return 10
    '\n    Ensure that orchestration produces the correct output with salt.function.\n\n    A salt execution module function does not return highstate data, so we\n    should not try to recursively output it as such.\n    The outlier to this rule is state.apply, but that is handled by the salt.state.\n\n    See https://github.com/saltstack/salt/issues/60029 for more detail.\n    '
    sls_contents = '\n    arg_clean_test:\n      salt.function:\n        - name: test.arg_clean\n        - arg:\n          - B flat major\n          - has 2 flats\n        - tgt: {minion_id}\n\n    ping_test:\n      salt.function:\n        - name: test.ping\n        - tgt: {minion_id}\n    '.format(minion_id=salt_minion.id)
    with salt_master.state_tree.base.temp_file('orch-function-test.sls', sls_contents):
        ret = salt_run_cli.run('--out=highstate', 'state.orchestrate', 'orch-function-test')
        assert ret.returncode == 0
        ret_output = [line.strip() for line in ret.stdout.splitlines()]
        assert 'args:' in ret_output
        assert '- B flat major' in ret_output
        assert '- has 2 flats' in ret_output
        assert 'True' in ret_output

def test_orchestrate_nested(salt_run_cli, salt_minion, salt_master, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    test salt-run state.orchestrate and failhard with nested orchestration\n    '
    testfile = tmp_path / 'ewu-2016-12-13'
    inner_sls = '\n    cmd.run:\n      salt.function:\n        - tgt: {}\n        - arg:\n          - {}\n        - failhard: True\n    '.format(salt_minion.id, pytest.helpers.shell_test_false())
    outer_sls = '\n    state.orchestrate:\n      salt.runner:\n        - mods: nested.inner\n        - failhard: True\n\n    cmd.run:\n      salt.function:\n        - tgt: {}\n        - arg:\n          - touch {}\n    '.format(salt_minion.id, testfile)
    with salt_master.state_tree.base.temp_file('nested/inner.sls', inner_sls), salt_master.state_tree.base.temp_file('nested/outer.sls', outer_sls):
        ret = salt_run_cli.run('state.orchestrate', 'nested.outer')
        assert ret.returncode != 0
        assert testfile.exists() is False

def test_orchestrate_with_mine(salt_run_cli, salt_minion, salt_master):
    if False:
        i = 10
        return i + 15
    '\n    test salt-run state.orchestrate with mine.get call in sls\n    '
    sls_contents = "\n    {% set minion = '" + salt_minion.id + '\' %}\n    {% set mine = salt.saltutil.runner(\'mine.get\', tgt=minion, fun=\'test.ping\') %}\n\n    {% if mine %}\n    test.ping:\n      salt.function:\n        - tgt: "{{ minion }}"\n    {% endif %}\n    '
    ret = salt_run_cli.run('mine.update', salt_minion.id)
    assert ret.returncode == 0
    with salt_master.state_tree.base.temp_file('orch/mine.sls', sls_contents):
        ret = salt_run_cli.run('state.orchestrate', 'orch.mine')
        assert ret.returncode == 0
        assert ret.data
        assert ret.data['data'][salt_master.id]
        for state_data in ret.data['data'][salt_master.id].values():
            assert state_data['changes']['ret']
            assert state_data['changes']['ret'][salt_minion.id] is True

def test_orchestrate_state_and_function_failure(salt_run_cli, salt_master, salt_minion):
    if False:
        print('Hello World!')
    '\n    Ensure that returns from failed minions are in the changes dict where\n    they belong, so they can be programmatically analyzed.\n\n    See https://github.com/saltstack/salt/issues/43204\n    '
    init_sls = '\n    Step01:\n      salt.state:\n        - tgt: {minion_id}\n        - sls:\n          - orch.issue43204.fail_with_changes\n\n    Step02:\n      salt.function:\n        - name: runtests_helpers.nonzero_retcode_return_false\n        - tgt: {minion_id}\n        - fail_function: runtests_helpers.fail_function\n    '.format(minion_id=salt_minion.id)
    fail_sls = '\n    test fail with changes:\n      test.fail_with_changes\n    '
    with salt_master.state_tree.base.temp_file('orch/issue43204/init.sls', init_sls), salt_master.state_tree.base.temp_file('orch/issue43204/fail_with_changes.sls', fail_sls):
        ret = salt_run_cli.run('saltutil.sync_modules')
        assert ret.returncode == 0
        ret = salt_run_cli.run('state.orchestrate', 'orch.issue43204')
        assert ret.returncode != 0
    data = ret.data['data'][salt_master.id]
    state_ret = data['salt_|-Step01_|-Step01_|-state']['changes']
    func_ret = data['salt_|-Step02_|-runtests_helpers.nonzero_retcode_return_false_|-function']['changes']
    for item in ('duration', 'start_time'):
        state_ret['ret'][salt_minion.id]['test_|-test fail with changes_|-test fail with changes_|-fail_with_changes'].pop(item)
    expected = {'out': 'highstate', 'ret': {salt_minion.id: {'test_|-test fail with changes_|-test fail with changes_|-fail_with_changes': {'__id__': 'test fail with changes', '__run_num__': 0, '__sls__': 'orch.issue43204.fail_with_changes', 'changes': {'testing': {'new': 'Something pretended to change', 'old': 'Unchanged'}}, 'comment': 'Failure!', 'name': 'test fail with changes', 'result': False}}}}
    assert state_ret == expected
    assert func_ret == {'ret': {salt_minion.id: False}}

def test_orchestrate_salt_function_return_false_failure(salt_run_cli, salt_minion, salt_master):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that functions that only return False in the return\n    are flagged as failed when run as orchestrations.\n\n    See https://github.com/saltstack/salt/issues/30367\n    '
    sls_contents = '\n    deploy_check:\n      salt.function:\n        - name: test.false\n        - tgt: {}\n    '.format(salt_minion.id)
    with salt_master.state_tree.base.temp_file('orch/issue30367.sls', sls_contents):
        ret = salt_run_cli.run('saltutil.sync_modules')
        assert ret.returncode == 0
        ret = salt_run_cli.run('state.orchestrate', 'orch.issue30367')
        assert ret.returncode != 0
    data = ret.data['data'][salt_master.id]
    state_result = data['salt_|-deploy_check_|-test.false_|-function']['result']
    func_ret = data['salt_|-deploy_check_|-test.false_|-function']['changes']
    assert state_result is False
    assert func_ret == {'ret': {salt_minion.id: False}}

def test_orchestrate_target_exists(salt_run_cli, salt_minion, salt_master):
    if False:
        print('Hello World!')
    '\n    test orchestration when target exists while using multiple states\n    '
    sls_contents = "\n    core:\n      salt.state:\n        - tgt: '{minion_id}*'\n        - sls:\n          - core\n\n    test-state:\n      salt.state:\n        - tgt: '{minion_id}*'\n        - sls:\n          - orch.target-test\n\n    cmd.run:\n      salt.function:\n        - tgt: '{minion_id}*'\n        - arg:\n          - echo test\n    ".format(minion_id=salt_minion.id)
    target_test_sls = '\n    always_true:\n      test.succeed_without_changes\n    '
    with salt_master.state_tree.base.temp_file('orch/target-exists.sls', sls_contents), salt_master.state_tree.base.temp_file('orch/target-test.sls', target_test_sls), salt_master.state_tree.base.temp_file('core.sls', target_test_sls):
        ret = salt_run_cli.run('state.orchestrate', 'orch.target-exists')
        assert ret.returncode == 0
        assert ret.data
    data = ret.data['data'][salt_master.id]
    to_check = {'core', 'test-state', 'cmd.run'}
    for state_data in data.values():
        if state_data['name'] == 'core':
            to_check.remove('core')
            assert state_data['result'] is True
        if state_data['name'] == 'test-state':
            assert state_data['result'] is True
            to_check.remove('test-state')
        if state_data['name'] == 'cmd.run':
            assert state_data['changes'] == {'ret': {salt_minion.id: 'test'}}
            to_check.remove('cmd.run')
    assert not to_check

def test_orchestrate_target_does_not_exist(salt_run_cli, salt_minion, salt_master):
    if False:
        print('Hello World!')
    '\n    test orchestration when target does not exist while using multiple states\n    '
    sls_contents = "\n    core:\n      salt.state:\n        - tgt: 'does-not-exist*'\n        - sls:\n          - core\n\n    test-state:\n      salt.state:\n        - tgt: '{minion_id}*'\n        - sls:\n          - orch.target-test\n\n    cmd.run:\n      salt.function:\n        - tgt: '{minion_id}*'\n        - arg:\n          - echo test\n    ".format(minion_id=salt_minion.id)
    target_test_sls = '\n    always_true:\n      test.succeed_without_changes\n    '
    with salt_master.state_tree.base.temp_file('orch/target-does-not-exist.sls', sls_contents), salt_master.state_tree.base.temp_file('orch/target-test.sls', target_test_sls), salt_master.state_tree.base.temp_file('core.sls', target_test_sls):
        ret = salt_run_cli.run('state.orchestrate', 'orch.target-does-not-exist')
        assert ret.returncode != 0
        assert ret.data
    data = ret.data['data'][salt_master.id]
    to_check = {'core', 'test-state', 'cmd.run'}
    for state_data in data.values():
        if state_data['name'] == 'core':
            to_check.remove('core')
            assert state_data['result'] is False
            assert state_data['comment'] == 'No minions returned'
        if state_data['name'] == 'test-state':
            assert state_data['result'] is True
            to_check.remove('test-state')
        if state_data['name'] == 'cmd.run':
            assert state_data['changes'] == {'ret': {salt_minion.id: 'test'}}
            to_check.remove('cmd.run')
    assert not to_check

def test_orchestrate_retcode(salt_run_cli, salt_master):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test orchestration with nonzero retcode set in __context__\n    '
    sls_contents = '\n    test_runner_success:\n      salt.runner:\n        - name: runtests_helpers.success\n\n    test_runner_failure:\n      salt.runner:\n        - name: runtests_helpers.failure\n\n    test_wheel_success:\n      salt.wheel:\n        - name: runtests_helpers.success\n\n    test_wheel_failure:\n      salt.wheel:\n        - name: runtests_helpers.failure\n    '
    with salt_master.state_tree.base.temp_file('orch/retcode.sls', sls_contents):
        ret = salt_run_cli.run('saltutil.sync_runners')
        assert ret.returncode == 0
        ret = salt_run_cli.run('saltutil.sync_wheel')
        assert ret.returncode == 0
        ret = salt_run_cli.run('state.orchestrate', 'orch.retcode')
        assert ret.returncode != 0
        assert ret.data
    data = ret.data['data'][salt_master.id]
    to_check = {'test_runner_success', 'test_runner_failure', 'test_wheel_failure', 'test_wheel_success'}
    for state_data in data.values():
        name = state_data['__id__']
        to_check.remove(name)
        if name in ('test_runner_success', 'test_wheel_success'):
            assert state_data['result'] is True
        if name in ('test_runner_failure', 'test_wheel_failure'):
            assert state_data['result'] is False
    assert not to_check

def test_orchestrate_batch_with_failhard_error(salt_run_cli, salt_master, salt_minion, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    test orchestration properly stops with failhard and batch.\n    '
    testfile = tmp_path / 'test-file'
    sls_contents = '\n    call_fail_state:\n      salt.state:\n        - tgt: {}\n        - batch: 1\n        - failhard: True\n        - sls: fail\n    '.format(salt_minion.id)
    fail_sls = '\n    {}:\n      file.managed:\n        - source: salt://hnlcfsdjhkzkdhynclarkhmcls\n    '.format(testfile)
    with salt_master.state_tree.base.temp_file('orch/batch.sls', sls_contents), salt_master.state_tree.base.temp_file('fail.sls', fail_sls):
        ret = salt_run_cli.run('state.orchestrate', 'orch.batch')
        assert ret.returncode != 0
    data = ret.data['data'][salt_master.id]
    result = data['salt_|-call_fail_state_|-call_fail_state_|-state']['result']
    changes = data['salt_|-call_fail_state_|-call_fail_state_|-state']['changes']
    assert result is False
    assert len(changes['ret']) == 1

def _check_skip(grains):
    if False:
        i = 10
        return i + 15
    if grains['os'] == 'Fedora':
        return True
    if grains['os'] == 'VMware Photon OS' and grains['osmajorrelease'] == 4:
        return True
    if grains['os'] == 'Ubuntu' and grains['osmajorrelease'] in (20, 22):
        return True
    return False

@pytest.mark.skip_initial_gh_actions_failure(skip=_check_skip)
def test_orchestrate_subset(salt_run_cli, salt_master, salt_minion, salt_sub_minion):
    if False:
        print('Hello World!')
    '\n    test orchestration state using subset\n    '
    sls_contents = "\n    test subset:\n      salt.state:\n        - tgt: '*minion*'\n        - subset: 1\n        - sls: test\n    "
    test_sls = '\n    test state:\n      test.succeed_without_changes:\n        - name: test\n    '
    with salt_master.state_tree.base.temp_file('orch/subset.sls', sls_contents), salt_master.state_tree.base.temp_file('test.sls', test_sls):
        ret = salt_run_cli.run('state.orchestrate', 'orch.subset', _timeout=60)
        assert ret.returncode == 0
    for state_data in ret.data['data'][salt_master.id].values():
        comment = state_data['comment']
        if salt_minion.id in comment:
            assert salt_sub_minion.id not in comment
        elif salt_sub_minion.id in comment:
            assert salt_minion.id not in comment
        else:
            pytest.fail("None of the targeted minions({}) show up in comment: '{}'".format(', '.join([salt_minion.id, salt_sub_minion.id]), comment))