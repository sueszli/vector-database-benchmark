import time
import pytest
pytestmark = [pytest.mark.core_test]

def test_minion_hangs_on_master_failure_50814(event_listener, salt_mm_master_1, salt_mm_master_2, salt_mm_minion_1, mm_master_2_salt_cli):
    if False:
        print('Hello World!')
    '\n    Check minion handling events for the alive master when another master is dead.\n    The case being checked here is described in details in issue #50814.\n    '
    event_count = 3
    while True:
        check_event_start_time = time.time()
        event_tag = f'myco/foo/bar/{event_count}'
        ret = mm_master_2_salt_cli.run('event.send', event_tag, minion_tgt=salt_mm_minion_1.id)
        assert ret.returncode == 0
        assert ret.data is True
        expected_patterns = [(salt_mm_master_1.id, event_tag), (salt_mm_master_2.id, event_tag)]
        matched_events = event_listener.wait_for_events(expected_patterns, after_time=check_event_start_time, timeout=30)
        assert matched_events.found_all_events, 'Minion is not responding to the second master after the first one has gone. Check #50814 for details.'
        event_count -= 1
        if event_count <= 0:
            break
        time.sleep(0.5)

    def wait_for_minion(salt_cli, tgt, timeout=60):
        if False:
            i = 10
            return i + 15
        start = time.time()
        while True:
            ret = salt_cli.run('test.ping', '--timeout=5', minion_tgt=tgt, _timeout=timeout)
            if ret.returncode == 0 and ret.data is True:
                break
            if time.time() - start > timeout:
                raise TimeoutError('Minion failed to respond top ping after timeout')
    salt_mm_master_1.after_start(wait_for_minion, salt_mm_master_1.salt_cli(), salt_mm_minion_1.id)
    with salt_mm_master_1.stopped():
        assert salt_mm_master_1.is_running() is False
        event_count = 1
        while True:
            check_event_start_time = time.time()
            event_tag = f'myco/foo/bar/{event_count}'
            ret = mm_master_2_salt_cli.run('event.send', event_tag, minion_tgt=salt_mm_minion_1.id)
            assert ret.returncode == 0
            assert ret.data is True
            expected_patterns = [(salt_mm_master_2.id, event_tag)]
            matched_events = event_listener.wait_for_events(expected_patterns, after_time=check_event_start_time, timeout=30)
            assert matched_events.found_all_events, 'Minion is not responding to the second master(events sent: {}) after the first has gone offline. Check #50814 for details.'.format(event_count)
            event_count += 1
            if event_count > 3:
                break
            time.sleep(0.5)