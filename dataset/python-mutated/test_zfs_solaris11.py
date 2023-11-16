"""
Tests for salt.modules.zfs on Solaris
"""
import pytest
import salt.loader
import salt.modules.zfs as zfs
import salt.utils.zfs
from tests.support.mock import MagicMock, patch
from tests.support.zfs import ZFSMockData

@pytest.fixture
def utils_patch():
    if False:
        for i in range(10):
            print('nop')
    return ZFSMockData().get_patched_utils()

@pytest.fixture
def configure_loader_modules(minion_opts):
    if False:
        return 10
    utils = salt.loader.utils(minion_opts, whitelist=['zfs'])
    zfs_obj = {zfs: {'__opts__': minion_opts, '__grains__': {'osarch': 'sparcv9', 'os_family': 'Solaris', 'osmajorrelease': 11, 'kernel': 'SunOS', 'kernelrelease': 5.11}, '__utils__': utils}}
    return zfs_obj

@pytest.mark.skip_unless_on_sunos
def test_get_success_solaris():
    if False:
        return 10
    '\n    Tests zfs get success\n    '
    cmd_out = {'pid': 7278, 'retcode': 0, 'stdout': 'testpool\tmountpoint\t/testpool\tdefault', 'stderr': ''}
    run_all_mock = MagicMock(return_value=cmd_out)
    patches = {'cmd.run_all': run_all_mock}
    with patch.dict(zfs.__salt__, patches):
        with patch('sys.platform', MagicMock(return_value='sunos5')):
            result = zfs.get('testpool', type='filesystem', properties='mountpoint')
            assert result == {'testpool': {'mountpoint': {'value': '/testpool', 'source': 'default'}}}
    run_all_mock.assert_called_once_with('/usr/sbin/zfs get -H -o name,property,value,source mountpoint testpool', python_shell=False)