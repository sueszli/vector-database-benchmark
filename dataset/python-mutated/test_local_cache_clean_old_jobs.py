"""
Unit tests for the Default Job Cache (local_cache).
"""
import os
import time
import pytest
import salt.returners.local_cache as local_cache
import salt.utils.files
import salt.utils.jid
import salt.utils.job
import salt.utils.platform
from tests.support.mock import MagicMock, patch

@pytest.fixture
def tmp_cache_dir(tmp_path):
    if False:
        i = 10
        return i + 15
    return tmp_path / 'cache_dir'

@pytest.fixture
def tmp_jid_dir(tmp_cache_dir):
    if False:
        return 10
    return tmp_cache_dir / 'jobs'

@pytest.fixture
def configure_loader_modules(tmp_cache_dir):
    if False:
        for i in range(10):
            print('nop')
    return {local_cache: {'__opts__': {'cachedir': str(tmp_cache_dir), 'keep_jobs_seconds': 3600}}}

@pytest.fixture
def make_tmp_jid_dirs(tmp_jid_dir):
    if False:
        for i in range(10):
            print('nop')

    def _make_tmp_jid_dirs(create_files=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Helper function to set up temporary directories and files used for\n        testing the clean_old_jobs function.\n\n        This emulates salt.utils.jid.jid_dir() by creating this structure:\n\n        RUNTIME_VARS.TMP_JID_DIR dir/\n            random dir from tempfile.mkdtemp/\n            'jid' directory/\n                'jid' file\n\n        Returns a temp_dir name and a jid_file_path. If create_files is False,\n        the jid_file_path will be None.\n        "
        tmp_jid_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = tmp_jid_dir / 'tmp_dir'
        temp_dir.mkdir(parents=True, exist_ok=True)
        jid_file_path = None
        if create_files:
            dir_name = temp_dir / 'jid'
            dir_name.mkdir(parents=True, exist_ok=True)
            jid_file_path = dir_name / 'jid'
            jid_file_path.write_text('this is a jid file')
            jid_file_path = str(jid_file_path)
        return (str(temp_dir), jid_file_path)
    return _make_tmp_jid_dirs

def test_clean_old_jobs_no_jid_root():
    if False:
        print('Hello World!')
    '\n    Tests that the function returns None when no jid_root is found.\n    '
    with patch('os.path.exists', MagicMock(return_value=False)):
        assert local_cache.clean_old_jobs() is None

def test_clean_old_jobs_empty_jid_dir_removed(make_tmp_jid_dirs, tmp_jid_dir):
    if False:
        i = 10
        return i + 15
    '\n    Tests that an empty JID dir is removed when it is old enough to be deleted.\n    '
    (jid_dir, jid_file) = make_tmp_jid_dirs(create_files=False)
    if salt.utils.platform.is_windows():
        time.sleep(0.01)
    assert jid_file is None
    with patch.dict(local_cache.__opts__, {'keep_jobs_seconds': 1e-08}):
        if salt.utils.platform.is_windows():
            time.sleep(0.25)
        local_cache.clean_old_jobs()
    assert [] == os.listdir(tmp_jid_dir)

def test_clean_old_jobs_empty_jid_dir_remains(make_tmp_jid_dirs, tmp_jid_dir):
    if False:
        i = 10
        return i + 15
    '\n    Tests that an empty JID dir is NOT removed because it was created within\n    the keep_jobs_seconds time frame.\n    '
    (jid_dir, jid_file) = make_tmp_jid_dirs(create_files=False)
    assert jid_file is None
    local_cache.clean_old_jobs()
    if salt.utils.platform.is_windows():
        jid_dir_name = jid_dir.rpartition('\\')[2]
    else:
        jid_dir_name = jid_dir.rpartition('/')[2]
    assert [jid_dir_name] == os.listdir(tmp_jid_dir)

def test_clean_old_jobs_jid_file_corrupted(make_tmp_jid_dirs, tmp_jid_dir):
    if False:
        return 10
    '\n    Tests that the entire JID dir is removed when the jid_file is not a file.\n    This scenario indicates a corrupted cache entry, so the entire dir is scrubbed.\n    '
    (jid_dir, jid_file) = make_tmp_jid_dirs()
    jid_dir_name = jid_file.rpartition(os.sep)[2]
    assert jid_dir_name == 'jid'
    with patch('os.path.isfile', MagicMock(return_value=False)) as mock:
        local_cache.clean_old_jobs()
    assert 1 == len(os.listdir(tmp_jid_dir))
    assert os.path.exists(jid_dir) is True
    assert os.path.isdir(jid_dir) is True
    assert os.path.exists(jid_dir_name) is False

def test_clean_old_jobs_jid_file_is_cleaned(make_tmp_jid_dirs, tmp_jid_dir):
    if False:
        print('Hello World!')
    '\n    Test that the entire JID dir is removed when a job is old enough to be removed.\n    '
    (jid_dir, jid_file) = make_tmp_jid_dirs()
    if salt.utils.platform.is_windows():
        time.sleep(0.01)
    jid_dir_name = jid_file.rpartition(os.sep)[2]
    assert jid_dir_name == 'jid'
    with patch.dict(local_cache.__opts__, {'keep_jobs_seconds': 1e-08}):
        if salt.utils.platform.is_windows():
            time.sleep(0.25)
        local_cache.clean_old_jobs()
    assert 1 == len(os.listdir(tmp_jid_dir))
    assert os.path.exists(jid_dir) is True
    assert os.path.isdir(jid_dir) is True
    assert os.path.exists(jid_dir_name) is False