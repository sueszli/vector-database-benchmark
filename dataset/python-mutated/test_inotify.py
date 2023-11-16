import logging
import os
import pytest
import salt.utils.files
from salt.beacons import inotify
try:
    import pyinotify
    HAS_PYINOTIFY = True
except ImportError:
    HAS_PYINOTIFY = False
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.skipif(HAS_PYINOTIFY is False, reason='pyinotify is not available.')]

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {inotify: {}}

@pytest.fixture(autouse=True)
def _close_inotify(configure_loader_modules):
    if False:
        while True:
            i = 10
    try:
        yield
    finally:
        inotify.close({})

def test_non_list_config():
    if False:
        while True:
            i = 10
    config = {}
    ret = inotify.validate(config)
    assert ret == (False, 'Configuration for inotify beacon must be a list.')

def test_empty_config():
    if False:
        for i in range(10):
            print('nop')
    config = [{}]
    ret = inotify.validate(config)
    _expected = (False, 'Configuration for inotify beacon must include files.')
    assert ret == _expected

def test_files_none_config():
    if False:
        while True:
            i = 10
    config = [{'files': None}]
    ret = inotify.validate(config)
    _expected = (False, 'Configuration for inotify beacon invalid, files must be a dict.')
    assert ret == _expected

def test_files_list_config():
    if False:
        print('Hello World!')
    config = [{'files': [{'/importantfile': {'mask': ['modify']}}]}]
    ret = inotify.validate(config)
    _expected = (False, 'Configuration for inotify beacon invalid, files must be a dict.')
    assert ret == _expected

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_file_open():
    if False:
        while True:
            i = 10
    path = os.path.realpath(__file__)
    config = [{'files': {path: {'mask': ['open']}}}]
    ret = inotify.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    ret = inotify.beacon(config)
    assert ret == []
    with salt.utils.files.fopen(path, 'r') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == path
    assert ret[0]['change'] == 'IN_OPEN'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_dir_no_auto_add(tmp_path):
    if False:
        while True:
            i = 10
    config = [{'files': {str(tmp_path): {'mask': ['create']}}}]
    ret = inotify.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    ret = inotify.beacon(config)
    assert ret == []
    fp = str(tmp_path / 'tmpfile')
    with salt.utils.files.fopen(fp, 'w') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_CREATE'
    with salt.utils.files.fopen(fp, 'r') as f:
        pass
    ret = inotify.beacon(config)
    assert ret == []

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_dir_auto_add(tmp_path):
    if False:
        while True:
            i = 10
    config = [{'files': {str(tmp_path): {'mask': ['create', 'open'], 'auto_add': True}}}]
    ret = inotify.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    ret = inotify.beacon(config)
    assert ret == []
    fp = str(tmp_path / 'tmpfile')
    with salt.utils.files.fopen(fp, 'w') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 2
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_CREATE'
    assert ret[1]['path'] == fp
    assert ret[1]['change'] == 'IN_OPEN'
    with salt.utils.files.fopen(fp, 'r') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_OPEN'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_dir_recurse(tmp_path):
    if False:
        return 10
    dp1 = str(tmp_path / 'subdir1')
    os.mkdir(dp1)
    dp2 = os.path.join(dp1, 'subdir2')
    os.mkdir(dp2)
    fp = os.path.join(dp2, 'tmpfile')
    with salt.utils.files.fopen(fp, 'w') as f:
        pass
    config = [{'files': {str(tmp_path): {'mask': ['open'], 'recurse': True}}}]
    ret = inotify.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    ret = inotify.beacon(config)
    assert ret == []
    with salt.utils.files.fopen(fp) as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 3
    assert ret[0]['path'] == dp1
    assert ret[0]['change'] == 'IN_OPEN|IN_ISDIR'
    assert ret[1]['path'] == dp2
    assert ret[1]['change'] == 'IN_OPEN|IN_ISDIR'
    assert ret[2]['path'] == fp
    assert ret[2]['change'] == 'IN_OPEN'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_dir_recurse_auto_add(tmp_path):
    if False:
        i = 10
        return i + 15
    dp1 = str(tmp_path / 'subdir1')
    os.mkdir(dp1)
    config = [{'files': {str(tmp_path): {'mask': ['create', 'delete'], 'recurse': True, 'auto_add': True}}}]
    ret = inotify.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    ret = inotify.beacon(config)
    assert ret == []
    dp2 = os.path.join(dp1, 'subdir2')
    os.mkdir(dp2)
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == dp2
    assert ret[0]['change'] == 'IN_CREATE|IN_ISDIR'
    fp = os.path.join(dp2, 'tmpfile')
    with salt.utils.files.fopen(fp, 'w') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_CREATE'
    os.remove(fp)
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_DELETE'

@pytest.mark.skip_on_freebsd(reason='Skip on FreeBSD - does not yet have full inotify/watchdog support')
def test_multi_files_exclude(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    dp1 = str(tmp_path / 'subdir1')
    dp2 = str(tmp_path / 'subdir2')
    os.mkdir(dp1)
    os.mkdir(dp2)
    _exclude1 = f'{str(tmp_path)}/subdir1/*tmpfile*$'
    _exclude2 = f'{str(tmp_path)}/subdir2/*filetmp*$'
    config = [{'files': {dp1: {'mask': ['create', 'delete'], 'recurse': True, 'exclude': [{_exclude1: {'regex': True}}], 'auto_add': True}}}, {'files': {dp2: {'mask': ['create', 'delete'], 'recurse': True, 'exclude': [{_exclude2: {'regex': True}}], 'auto_add': True}}}]
    ret = inotify.validate(config)
    assert ret == (True, 'Valid beacon configuration')
    fp = os.path.join(dp1, 'tmpfile')
    with salt.utils.files.fopen(fp, 'w') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 0
    os.remove(fp)
    ret = inotify.beacon(config)
    assert len(ret) == 0
    fp = os.path.join(dp2, 'tmpfile')
    with salt.utils.files.fopen(fp, 'w') as f:
        pass
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_CREATE'
    os.remove(fp)
    ret = inotify.beacon(config)
    assert len(ret) == 1
    assert ret[0]['path'] == fp
    assert ret[0]['change'] == 'IN_DELETE'

def test__get_notifier():
    if False:
        return 10
    config = {'files': {'/tmp/httpd/vhost.d': {'mask': ['delete', 'modify'], 'recurse': True, 'auto_add': True, 'exclude': [{'/tmp/httpd/vhost.d/.+?\\.sw[px]*$|4913|~$': {'regex': True}}]}, '/tmp/httpd/conf.d': {'mask': ['delete', 'modify'], 'recurse': True, 'auto_add': True, 'exclude': [{'/tmp/httpd/vhost.d/.+?\\.sw[px]*$|4913|~$': {'regex': True}}]}, '/tmp/httpd/conf': {'mask': ['delete', 'modify'], 'recurse': True, 'auto_add': True, 'exclude': [{'/tmp/httpd/vhost.d/.+?\\.sw[px]*$|4913|~$': {'regex': True}}]}}, 'coalesce': True, 'beacon_module': 'inotify', '_beacon_name': 'httpd.inotify'}
    ret = inotify._get_notifier(config)
    assert 'inotify.queue' in inotify.__context__
    assert 'httpd.inotify.notifier' in inotify.__context__