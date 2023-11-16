import logging
import os
import time
from typing import List, Optional
from freezegun import freeze_time
import pytest
import ray
import ray.cloudpickle as pickle
from ray.train._internal.storage import _upload_to_fs_path, _download_from_fs_path, get_fs_and_path, _FilesystemSyncer
from ray.train._internal.syncer import _BackgroundProcess
from ray.train.tests.test_new_persistence import _create_mock_custom_fs

@pytest.fixture
def propagate_logs():
    if False:
        return 10
    logger = logging.getLogger('ray')
    logger.propagate = True
    yield
    logger.propagate = False

@pytest.fixture
def ray_start_4_cpus():
    if False:
        while True:
            i = 10
    address_info = ray.init(num_cpus=4, configure_logging=False)
    yield address_info
    ray.shutdown()

@pytest.fixture
def ray_start_2_cpus():
    if False:
        return 10
    address_info = ray.init(num_cpus=2, configure_logging=False)
    yield address_info
    ray.shutdown()

@pytest.fixture
def shutdown_only():
    if False:
        i = 10
        return i + 15
    yield None
    ray.shutdown()

@pytest.fixture
def temp_data_dirs(tmp_path):
    if False:
        return 10
    tmp_source = tmp_path / 'source'
    tmp_target = tmp_path / 'target'
    tmp_target.mkdir()
    os.makedirs(os.path.join(tmp_source, 'subdir', 'nested'))
    os.makedirs(os.path.join(tmp_source, 'subdir_exclude', 'something'))
    files = ['level0.txt', 'level0_exclude.txt', 'subdir/level1.txt', 'subdir/level1_exclude.txt', 'subdir/nested/level2.txt', 'subdir_nested_level2_exclude.txt', 'subdir_exclude/something/somewhere.txt']
    for file in files:
        with open(os.path.join(tmp_source, file), 'w') as f:
            f.write('Data')
    yield (str(tmp_source), str(tmp_target))

@pytest.fixture
def syncer(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    yield _FilesystemSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path))

def assert_file(exists: bool, root: str, path: str):
    if False:
        while True:
            i = 10
    full_path = os.path.join(root, path)
    if exists:
        assert os.path.exists(full_path)
    else:
        assert not os.path.exists(full_path)

def test_syncer_sync_up(temp_data_dirs, syncer):
    if False:
        i = 10
        return i + 15
    'Check that syncing up works'
    (tmp_source, tmp_target) = temp_data_dirs
    syncer.sync_up(local_dir=tmp_source, remote_dir='/test/test_syncer_sync_up_down')
    syncer.wait()
    _download_from_fs_path(syncer.storage_filesystem, '/test/test_syncer_sync_up_down', tmp_target)
    assert_file(True, tmp_target, 'level0.txt')
    assert_file(True, tmp_target, 'level0_exclude.txt')
    assert_file(True, tmp_target, 'subdir/level1.txt')
    assert_file(True, tmp_target, 'subdir/level1_exclude.txt')
    assert_file(True, tmp_target, 'subdir/nested/level2.txt')
    assert_file(True, tmp_target, 'subdir_nested_level2_exclude.txt')
    assert_file(True, tmp_target, 'subdir_exclude/something/somewhere.txt')

def test_syncer_sync_exclude(temp_data_dirs, syncer):
    if False:
        for i in range(10):
            print('nop')
    'Check that the exclude parameter works'
    (tmp_source, tmp_target) = temp_data_dirs
    syncer.sync_up(local_dir=tmp_source, remote_dir='/test/test_syncer_sync_exclude', exclude=['*_exclude*'])
    syncer.wait()
    _download_from_fs_path(syncer.storage_filesystem, '/test/test_syncer_sync_exclude', tmp_target)
    assert_file(True, tmp_target, 'level0.txt')
    assert_file(False, tmp_target, 'level0_exclude.txt')
    assert_file(True, tmp_target, 'subdir/level1.txt')
    assert_file(False, tmp_target, 'subdir/level1_exclude.txt')
    assert_file(True, tmp_target, 'subdir/nested/level2.txt')
    assert_file(False, tmp_target, 'subdir_nested_level2_exclude.txt')
    assert_file(False, tmp_target, 'subdir_exclude/something/somewhere.txt')

def test_sync_up_if_needed(temp_data_dirs, tmp_path):
    if False:
        while True:
            i = 10
    'Check that we only sync up again after sync period'
    (tmp_source, tmp_target) = temp_data_dirs
    with freeze_time() as frozen:
        syncer = _FilesystemSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60)
        assert syncer.sync_up_if_needed(local_dir=tmp_source, remote_dir='/test/test_sync_up_not_needed')
        syncer.wait()
        frozen.tick(30)
        assert not syncer.sync_up_if_needed(local_dir=tmp_source, remote_dir='/test/test_sync_up_not_needed')
        frozen.tick(30)
        assert syncer.sync_up_if_needed(local_dir=tmp_source, remote_dir='/test/test_sync_up_not_needed')

def test_syncer_still_running_no_sync(temp_data_dirs, tmp_path):
    if False:
        print('Hello World!')
    'Check that no new sync is issued if old sync is still running'
    (tmp_source, tmp_target) = temp_data_dirs

    class FakeSyncProcess:

        @property
        def is_running(self):
            if False:
                for i in range(10):
                    print('nop')
            return True

        @property
        def start_time(self):
            if False:
                print('Hello World!')
            return float('inf')
    syncer = _FilesystemSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60)
    syncer._sync_process = FakeSyncProcess()
    assert not syncer.sync_up_if_needed(local_dir=tmp_source, remote_dir='/test/test_syncer_still_running_no_sync')

def test_syncer_not_running_sync(temp_data_dirs, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Check that new sync is issued if old sync completed'
    (tmp_source, tmp_target) = temp_data_dirs

    class FakeSyncProcess:

        @property
        def is_running(self):
            if False:
                return 10
            return False

        def wait(self):
            if False:
                while True:
                    i = 10
            return True
    syncer = _FilesystemSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60)
    syncer._sync_process = FakeSyncProcess()
    assert syncer.sync_up_if_needed(local_dir=tmp_source, remote_dir='/test/test_syncer_not_running_sync')

def test_syncer_hanging_sync_with_timeout(temp_data_dirs, tmp_path):
    if False:
        i = 10
        return i + 15
    'Check that syncing times out when the sync process is hanging.'
    (tmp_source, tmp_target) = temp_data_dirs

    def _hanging_sync_up_command(*args, **kwargs):
        if False:
            print('Hello World!')
        time.sleep(200)

    class _HangingSyncer(_FilesystemSyncer):

        def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List]=None):
            if False:
                i = 10
                return i + 15
            return (_hanging_sync_up_command, {})
    syncer = _HangingSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60, sync_timeout=10)

    def sync_up():
        if False:
            while True:
                i = 10
        return syncer.sync_up(local_dir=tmp_source, remote_dir='/test/test_syncer_timeout')
    with freeze_time() as frozen:
        assert sync_up()
        frozen.tick(5)
        assert not sync_up()
        frozen.tick(5)
        assert sync_up()
        frozen.tick(20)
        with pytest.raises(TimeoutError):
            syncer.wait()

def test_syncer_not_running_sync_last_failed(propagate_logs, caplog, temp_data_dirs, tmp_path):
    if False:
        while True:
            i = 10
    'Check that new sync is issued if old sync completed'
    caplog.set_level(logging.WARNING)
    (tmp_source, tmp_target) = temp_data_dirs

    class FakeSyncProcess(_BackgroundProcess):

        @property
        def is_running(self):
            if False:
                i = 10
                return i + 15
            return False

        def wait(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            raise RuntimeError('Sync failed')
    syncer = _FilesystemSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60)
    syncer._sync_process = FakeSyncProcess(lambda : None)
    assert syncer.sync_up_if_needed(local_dir=tmp_source, remote_dir='/test/test_syncer_not_running_sync')
    assert 'Last sync command failed' in caplog.text

def test_syncer_wait_or_retry_failure(temp_data_dirs, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Check that the wait or retry API fails after max_retries.'
    (tmp_source, tmp_target) = temp_data_dirs
    syncer = _FilesystemSyncer(storage_filesystem=lambda : 'error', sync_period=60)
    syncer.sync_up(local_dir=tmp_source, remote_dir='/test/test_syncer_wait_or_retry')
    with pytest.raises(RuntimeError) as e:
        syncer.wait_or_retry(max_retries=3, backoff_s=0)
    assert 'Failed sync even after 3 retries.' in str(e.value)

def test_syncer_wait_or_retry_timeout(temp_data_dirs, tmp_path):
    if False:
        i = 10
        return i + 15
    'Check that the wait or retry API raises a timeout error after `sync_timeout`.'
    (tmp_source, tmp_target) = temp_data_dirs

    def slow_upload(*args, **kwargs):
        if False:
            print('Hello World!')
        time.sleep(5)

    class HangingSyncer(_FilesystemSyncer):

        def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List]=None):
            if False:
                print('Hello World!')
            return (slow_upload, dict(local_path=local_path, uri=uri, exclude=exclude))
    syncer = HangingSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60, sync_timeout=0.1)
    syncer.sync_up(local_dir=tmp_source, remote_dir='/test/timeout')
    with pytest.raises(RuntimeError) as e:
        syncer.wait_or_retry(max_retries=3, backoff_s=0)
        assert 'Failed sync even after 3 retries.' in str(e.value)
        assert isinstance(e.value.__cause__, TimeoutError)

def test_syncer_wait_or_retry_eventual_success(temp_data_dirs, tmp_path):
    if False:
        return 10
    'Check that the wait or retry API succeeds for a sync_down that\n    fails, times out, then succeeds.'
    (tmp_source, tmp_target) = temp_data_dirs
    success = tmp_path / 'success'
    fail_marker = tmp_path / 'fail_marker'
    hang_marker = tmp_path / 'hang_marker'

    def eventual_upload(*args, **kwargs):
        if False:
            while True:
                i = 10
        if not fail_marker.exists():
            fail_marker.write_text('.', encoding='utf-8')
            raise RuntimeError('Failing')
        elif not hang_marker.exists():
            hang_marker.write_text('.', encoding='utf-8')
            time.sleep(5)
        else:
            success.write_text('.', encoding='utf-8')

    class EventualSuccessSyncer(_FilesystemSyncer):

        def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List]=None):
            if False:
                return 10
            return (eventual_upload, dict(local_path=local_path, uri=uri, exclude=exclude))
    syncer = EventualSuccessSyncer(storage_filesystem=_create_mock_custom_fs(tmp_path), sync_period=60, sync_timeout=0.5)
    syncer.sync_up(local_dir=tmp_source, remote_dir='/test/eventual_success')
    syncer.wait_or_retry(max_retries=2, backoff_s=0)
    assert success.exists()

def test_syncer_serialize(temp_data_dirs, syncer):
    if False:
        while True:
            i = 10
    (tmp_source, tmp_target) = temp_data_dirs
    syncer.sync_up(local_dir=tmp_source, remote_dir='/test/serialize')
    serialized = pickle.dumps(syncer)
    loaded_syncer = pickle.loads(serialized)
    assert not loaded_syncer._sync_process

def test_sync_many_files_local_to_cloud(mock_s3_bucket_uri, tmp_path):
    if False:
        return 10
    source_dir = tmp_path / 'source'
    check_dir = tmp_path / 'check'
    source_dir.mkdir()
    check_dir.mkdir()
    for i in range(256):
        (source_dir / str(i)).write_text('', encoding='utf-8')
    (fs, fs_path) = get_fs_and_path(mock_s3_bucket_uri)
    _upload_to_fs_path(source_dir, fs, fs_path)
    _download_from_fs_path(fs, fs_path, check_dir)
    assert (check_dir / '255').exists()

def test_sync_many_files_local_to_local(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    (tmp_path / 'source').mkdir()
    for i in range(256):
        (tmp_path / 'source' / str(i)).write_text('', encoding='utf-8')
    (fs, fs_path) = get_fs_and_path(str(tmp_path / 'destination'))
    _upload_to_fs_path(str(tmp_path / 'source'), fs, fs_path)
    assert (tmp_path / 'destination' / '255').exists()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))