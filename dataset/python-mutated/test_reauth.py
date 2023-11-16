import logging
import os
import threading
import time
import pytest
pytestmark = [pytest.mark.slow_test, pytest.mark.windows_whitelisted]
log = logging.getLogger(__name__)

def minion_func(salt_minion, event_listener, salt_master, timeout):
    if False:
        for i in range(10):
            print('nop')
    start = time.time()
    with salt_minion.started(start_timeout=timeout * 2, max_start_attempts=1):
        new_start = time.time()
        while time.time() < new_start + timeout * 2:
            if event_listener.get_events([(salt_master.id, f'salt/job/*/ret/{salt_minion.id}')], after_time=start):
                break
            time.sleep(5)

@pytest.fixture(scope='module')
def timeout():
    if False:
        i = 10
        return i + 15
    return int(os.environ.get('SALT_CI_REAUTH_MASTER_WAIT', 150))

def test_reauth(salt_cli, salt_minion, salt_master, timeout, event_listener):
    if False:
        for i in range(10):
            print('nop')
    assert salt_cli.run('test.ping', minion_tgt=salt_minion.id).data is True
    salt_master.terminate()
    salt_minion.terminate()
    log.debug('Master and minion stopped for reauth test, waiting for %s seconds', timeout)
    log.debug('Restarting the reauth minion')
    minion_proc = threading.Thread(target=minion_func, args=(salt_minion, event_listener, salt_master, timeout))
    minion_proc.start()
    time.sleep(timeout)
    log.debug('Restarting the reauth master')
    start = time.time()
    salt_master.start()
    event_listener.wait_for_events([(salt_master.id, f'salt/minion/{salt_minion.id}/start')], after_time=start, timeout=timeout * 2)
    assert salt_cli.run('test.ping', minion_tgt=salt_minion.id).data is True
    minion_proc.join()