import time
import pytest
from saltfactories.utils import random_string

@pytest.fixture(scope='function')
def salt_minion_retry(salt_master_factory, salt_minion_id):
    if False:
        for i in range(10):
            print('nop')
    config_overrides = {'return_retry_timer_max': 0, 'return_retry_timer': 5, 'return_retry_tries': 30}
    factory = salt_master_factory.salt_minion_daemon(random_string('retry-minion-'), overrides=config_overrides, extra_cli_arguments_after_first_start_failure=['--log-level=info'])
    factory.after_terminate(pytest.helpers.remove_stale_minion_key, salt_master_factory, factory.id)
    with factory.started():
        yield factory

@pytest.mark.slow_test
def test_publish_retry(salt_master, salt_minion_retry, salt_cli, salt_run_cli):
    if False:
        while True:
            i = 10
    rtn = salt_cli.run('test.sleep', '10', '--async', minion_tgt=salt_minion_retry.id)
    jid = rtn.stdout.strip().split(' ')[-1]
    time.sleep(1)
    with salt_master.stopped():
        assert salt_run_cli.run('jobs.lookup_jid', jid, _timeout=10).data == {}
        time.sleep(5)
    data = None
    for i in range(1, 30):
        time.sleep(1)
        data = salt_run_cli.run('jobs.lookup_jid', jid, _timeout=60).data
        if data:
            break
    assert salt_minion_retry.id in data
    assert data[salt_minion_retry.id] is True

@pytest.mark.slow_test
def test_pillar_timeout(salt_master_factory):
    if False:
        i = 10
        return i + 15
    cmd = '\n    python -c "import time; time.sleep(2.5); print(\'{\\"foo\\": \\"bar\\"}\');"\n    '.strip()
    master_overrides = {'ext_pillar': [{'cmd_json': cmd}], 'auto_accept': True, 'worker_threads': 3, 'peer': True}
    minion_overrides = {'auth_timeout': 20, 'request_channel_timeout': 5, 'request_channel_tries': 1}
    sls_name = 'issue-50221'
    sls_contents = '\n    custom_test_state:\n      test.configurable_test_state:\n        - name: example\n        - changes: True\n        - result: True\n        - comment: "Nothing has acutally been changed"\n    '
    master = salt_master_factory.salt_master_daemon('pillar-timeout-master', overrides=master_overrides)
    minion1 = master.salt_minion_daemon(random_string('pillar-timeout-1-'), overrides=minion_overrides)
    minion2 = master.salt_minion_daemon(random_string('pillar-timeout-2-'), overrides=minion_overrides)
    minion3 = master.salt_minion_daemon(random_string('pillar-timeout-3-'), overrides=minion_overrides)
    minion4 = master.salt_minion_daemon(random_string('pillar-timeout-4-'), overrides=minion_overrides)
    cli = master.salt_cli()
    sls_tempfile = master.state_tree.base.temp_file(f'{sls_name}.sls', sls_contents)
    with master.started(), minion1.started(), minion2.started(), minion3.started(), minion4.started(), sls_tempfile:
        proc = cli.run('state.sls', sls_name, minion_tgt='*')
        assert proc.returncode == 1
        minion_timed_out = False
        for key in proc.data:
            if isinstance(proc.data[key], str):
                if 'Pillar timed out' in proc.data[key]:
                    minion_timed_out = True
                    break
        assert minion_timed_out is True