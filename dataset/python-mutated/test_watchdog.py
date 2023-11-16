import os
import time
import pytest
import salt.utils.files
import salt.utils.platform
from salt.beacons import watchdog
pytestmark = [pytest.mark.skipif(watchdog.HAS_WATCHDOG is False, reason='watchdog is not available'), pytest.mark.skipif(salt.utils.platform.is_darwin(), reason='Tests were being skipped pre macos under nox. Keep it like that for now.')]

def check_events(config):
    if False:
        return 10
    total_delay = 1
    delay_per_loop = 0.02
    for _ in range(int(total_delay / delay_per_loop)):
        events = watchdog.beacon(config)
        if events:
            return events
        time.sleep(delay_per_loop)
    return []

def create(path, content=None):
    if False:
        return 10
    with salt.utils.files.fopen(path, 'w') as f:
        if content:
            f.write(content)
        os.fsync(f)

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {watchdog: {}}

@pytest.fixture(autouse=True)
def _close_watchdog(configure_loader_modules):
    if False:
        return 10
    try:
        yield
    finally:
        watchdog.close({})

def assertValid(config):
    if False:
        for i in range(10):
            print('nop')
    ret = watchdog.validate(config)
    assert ret == (True, 'Valid beacon configuration')

def test_empty_config():
    if False:
        while True:
            i = 10
    config = [{}]
    ret = watchdog.beacon(config)
    assert ret == []

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_file_create(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    path = str(tmp_path / 'tmpfile')
    config = [{'directories': {str(tmp_path): {'mask': ['create']}}}]
    assertValid(config)
    assert watchdog.beacon(config) == []
    create(path)
    ret = check_events(config)
    assert len(ret) == 1
    assert ret[0]['path'] == path
    assert ret[0]['change'] == 'created'

def test_file_modified(tmp_path):
    if False:
        print('Hello World!')
    path = str(tmp_path / 'tmpfile')
    create(path)
    config = [{'directories': {str(tmp_path): {'mask': ['modify']}}}]
    assertValid(config)
    assert watchdog.beacon(config) == []
    create(path, 'some content')
    ret = check_events(config)
    modified = False
    for event in ret:
        if event['change'] == 'modified':
            if event['path'] == path:
                modified = True
    assert modified

def test_file_deleted(tmp_path):
    if False:
        print('Hello World!')
    path = str(tmp_path / 'tmpfile')
    create(path)
    config = [{'directories': {str(tmp_path): {'mask': ['delete']}}}]
    assertValid(config)
    assert watchdog.beacon(config) == []
    os.remove(path)
    ret = check_events(config)
    assert len(ret) == 1
    assert ret[0]['path'] == path
    assert ret[0]['change'] == 'deleted'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_file_moved(tmp_path):
    if False:
        return 10
    path = str(tmp_path / 'tmpfile')
    create(path)
    config = [{'directories': {str(tmp_path): {'mask': ['move']}}}]
    assertValid(config)
    assert watchdog.beacon(config) == []
    os.rename(path, path + '_moved')
    ret = check_events(config)
    assert len(ret) == 1
    assert ret[0]['path'] == path
    assert ret[0]['change'] == 'moved'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_file_create_in_directory(tmp_path):
    if False:
        while True:
            i = 10
    config = [{'directories': {str(tmp_path): {'mask': ['create']}}}]
    assertValid(config)
    assert watchdog.beacon(config) == []
    path = str(tmp_path / 'tmpfile')
    create(path)
    ret = check_events(config)
    assert len(ret) == 1
    assert ret[0]['path'] == path
    assert ret[0]['change'] == 'created'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
@pytest.mark.slow_test
def test_trigger_all_possible_events(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    path = str(tmp_path / 'tmpfile')
    moved = path + '_moved'
    config = [{'directories': {str(tmp_path): {}}}]
    assertValid(config)
    assert watchdog.beacon(config) == []
    create(path)
    create(path, 'modified content')
    os.rename(path, moved)
    os.remove(moved)
    time.sleep(1)
    ret = check_events(config)
    events = {'created': '', 'deleted': '', 'moved': ''}
    modified = False
    for event in ret:
        if event['change'] == 'created':
            assert event['path'] == path
            events.pop('created', '')
        if event['change'] == 'moved':
            assert event['path'] == path
            events.pop('moved', '')
        if event['change'] == 'deleted':
            assert event['path'] == moved
            events.pop('deleted', '')
        if event['change'] == 'modified':
            if event['path'] == path:
                modified = True
    assert modified
    assert events == {}