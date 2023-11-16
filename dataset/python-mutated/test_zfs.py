"""
Tests for salt.modules.zfs

:codeauthor:    Nitin Madhok <nmadhok@g.clemson.edu>, Jorge Schrauwen <sjorge@blackdot.be>
:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:depends:       salt.utils.zfs
:platform:      illumos,freebsd,linux
"""
import pytest
import salt.loader
import salt.modules.zfs as zfs
import salt.utils.zfs
from salt.utils.dateutils import strftime
from salt.utils.odict import OrderedDict
from tests.support.mock import MagicMock, patch
from tests.support.zfs import ZFSMockData
pytestmark = [pytest.mark.slow_test]

@pytest.fixture
def utils_patch():
    if False:
        for i in range(10):
            print('nop')
    return ZFSMockData().get_patched_utils()

@pytest.fixture
def configure_loader_modules(minion_opts):
    if False:
        i = 10
        return i + 15
    utils = salt.loader.utils(minion_opts, whitelist=['zfs', 'args', 'systemd', 'path', 'platform'])
    zfs_obj = {zfs: {'__opts__': minion_opts, '__utils__': utils}}
    return zfs_obj

def test_exists_success(utils_patch):
    if False:
        return 10
    '\n    Tests successful return of exists function\n    '
    ret = {}
    ret['stdout'] = 'NAME        USED  AVAIL  REFER  MOUNTPOINT\nmyzpool/mydataset    30K   157G    30K  /myzpool/mydataset'
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert zfs.exists('myzpool/mydataset')

def test_exists_failure_not_exists(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests unsuccessful return of exists function if dataset does not exist\n    '
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot open 'myzpool/mydataset': dataset does not exist"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert not zfs.exists('myzpool/mydataset')

def test_exists_failure_invalid_name(utils_patch):
    if False:
        return 10
    '\n    Tests unsuccessful return of exists function if dataset name is invalid\n    '
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot open 'myzpool/': invalid dataset name"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert not zfs.exists('myzpool/')

def test_create_success(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests successful return of create function on ZFS file system creation\n    '
    res = OrderedDict([('created', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool/mydataset')

def test_create_success_with_create_parent(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests successful return of create function when ``create_parent=True``\n    '
    res = OrderedDict([('created', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool/mydataset/mysubdataset', create_parent=True)

def test_create_success_with_properties(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests successful return of create function on ZFS file system creation (with properties)\n    '
    res = OrderedDict([('created', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool/mydataset', properties={'mountpoint': '/export/zfs', 'sharenfs': 'on'})

def test_create_error_missing_dataset(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests unsuccessful return of create function if dataset name is missing\n    '
    res = OrderedDict([('created', False), ('error', "cannot create 'myzpool': missing dataset name")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot create 'myzpool': missing dataset name"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool')

def test_create_error_trailing_slash(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests unsuccessful return of create function if trailing slash in name is present\n    '
    res = OrderedDict([('created', False), ('error', "cannot create 'myzpool/': trailing slash in name")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot create 'myzpool/': trailing slash in name"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool/')

def test_create_error_no_such_pool(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests unsuccessful return of create function if the pool is not present\n    '
    res = OrderedDict([('created', False), ('error', "cannot create 'myzpool/mydataset': no such pool 'myzpool'")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot create 'myzpool/mydataset': no such pool 'myzpool'"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool/mydataset')

def test_create_error_missing_parent(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests unsuccessful return of create function if the parent datasets do not exist\n    '
    res = OrderedDict([('created', False), ('error', "cannot create 'myzpool/mydataset/mysubdataset': parent does not exist")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot create 'myzpool/mydataset/mysubdataset': parent does not exist"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.create('myzpool/mydataset/mysubdataset')

def test_destroy_success(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests successful return of destroy function on ZFS file system destruction\n    '
    res = OrderedDict([('destroyed', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.destroy('myzpool/mydataset')

def test_destroy_error_not_exists(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests failure return of destroy function on ZFS file system destruction\n    '
    res = OrderedDict([('destroyed', False), ('error', "cannot open 'myzpool/mydataset': dataset does not exist")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot open 'myzpool/mydataset': dataset does not exist"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.destroy('myzpool/mydataset')

def test_destroy_error_has_children(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests failure return of destroy function on ZFS file system destruction\n    '
    res = OrderedDict([('destroyed', False), ('error', '\n'.join(["cannot destroy 'myzpool/mydataset': filesystem has children", "use 'recursive=True' to destroy the following datasets:", 'myzpool/mydataset@snapshot']))])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = '\n'.join(["cannot destroy 'myzpool/mydataset': filesystem has children", "use '-r' to destroy the following datasets:", 'myzpool/mydataset@snapshot'])
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.destroy('myzpool/mydataset')

@pytest.mark.slow_test
def test_rename_success(utils_patch):
    if False:
        return 10
    '\n    Tests successful return of rename function\n    '
    res = OrderedDict([('renamed', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.rename('myzpool/mydataset', 'myzpool/newdataset')

def test_rename_error_not_exists(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests failure return of rename function\n    '
    res = OrderedDict([('renamed', False), ('error', "cannot open 'myzpool/mydataset': dataset does not exist")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot open 'myzpool/mydataset': dataset does not exist"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.rename('myzpool/mydataset', 'myzpool/newdataset')

def test_list_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs list\n    '
    res = OrderedDict([('myzpool', OrderedDict([('used', 849329782784), ('avail', 1081258016768), ('refer', 98304), ('mountpoint', '/myzpool')]))])
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = 'myzpool\t791G\t1007G\t96K\t/myzpool'
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.list_('myzpool')

@pytest.mark.slow_test
def test_list_parsable_success(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs list with parsable set to False\n    '
    res = OrderedDict([('myzpool', OrderedDict([('used', '791G'), ('avail', '1007G'), ('refer', '96K'), ('mountpoint', '/myzpool')]))])
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = 'myzpool\t791G\t1007G\t96K\t/myzpool'
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.list_('myzpool', parsable=False)

def test_list_custom_success(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs list\n    '
    res = OrderedDict([('myzpool', OrderedDict([('canmount', True), ('used', 849329782784), ('avail', 1081258016768), ('compression', False)]))])
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = 'myzpool\ton\t791G\t1007G\toff'
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.list_('myzpool', properties='canmount,used,avail,compression')

def test_list_custom_parsable_success(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests zfs list\n    '
    res = OrderedDict([('myzpool', OrderedDict([('canmount', 'on'), ('used', '791G'), ('avail', '1007G'), ('compression', 'off')]))])
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = 'myzpool\ton\t791G\t1007G\toff'
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.list_('myzpool', properties='canmount,used,avail,compression', parsable=False)

def test_list_error_no_dataset(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs list\n    '
    res = OrderedDict()
    ret = {}
    ret['retcode'] = 1
    ret['stdout'] = "cannot open 'myzpool': dataset does not exist"
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.list_('myzpool')

@pytest.mark.slow_test
def test_list_mount_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs list_mount\n    '
    res = OrderedDict([('myzpool/data', '/data'), ('myzpool/data/ares', '/data/ares')])
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = '\n'.join(['myzpool/data\t\t\t\t/data', 'myzpool/data/ares\t\t\t/data/ares'])
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.list_mount()

def test_mount_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs mount of filesystem\n    '
    res = OrderedDict([('mounted', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.mount('myzpool/mydataset')

def test_mount_failure(utils_patch):
    if False:
        return 10
    '\n    Tests zfs mount of already mounted filesystem\n    '
    res = OrderedDict([('mounted', False), ('error', "cannot mount 'myzpool/mydataset': filesystem already mounted")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot mount 'myzpool/mydataset': filesystem already mounted"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.mount('myzpool/mydataset')

def test_unmount_success(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs unmount of filesystem\n    '
    res = OrderedDict([('unmounted', True)])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = ''
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.unmount('myzpool/mydataset')

def test_unmount_failure(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs unmount of already mounted filesystem\n    '
    res = OrderedDict([('unmounted', False), ('error', "cannot mount 'myzpool/mydataset': not currently mounted")])
    ret = {}
    ret['stdout'] = ''
    ret['stderr'] = "cannot mount 'myzpool/mydataset': not currently mounted"
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.unmount('myzpool/mydataset')

def test_inherit_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs inherit of compression property\n    '
    res = OrderedDict([('inherited', True)])
    ret = {'pid': 45193, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.inherit('compression', 'myzpool/mydataset')

def test_inherit_failure(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs inherit of canmount\n    '
    res = OrderedDict([('inherited', False), ('error', "'canmount' property cannot be inherited")])
    ret = {'pid': 43898, 'retcode': 1, 'stderr': "'canmount' property cannot be inherited", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.inherit('canmount', 'myzpool/mydataset')

@pytest.mark.slow_test
def test_diff(utils_patch):
    if False:
        return 10
    '\n    Tests zfs diff\n    '
    res = ['1517063879.144517494\tM\t\t/data/test/', '1517063875.296592355\t+\t\t/data/test/world', '1517063879.274438467\t+\t\t/data/test/hello']
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = '\n'.join(['1517063879.144517494\tM\t\t/data/test/', '1517063875.296592355\t+\t\t/data/test/world', '1517063879.274438467\t+\t\t/data/test/hello'])
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.diff('myzpool/mydataset@yesterday', 'myzpool/mydataset')

def test_diff_parsed_time(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs diff\n    '
    res = OrderedDict([(strftime(1517063879.1445174, '%Y-%m-%d.%H:%M:%S.%f'), 'M\t\t/data/test/'), (strftime(1517063875.2965925, '%Y-%m-%d.%H:%M:%S.%f'), '+\t\t/data/test/world'), (strftime(1517063879.2744384, '%Y-%m-%d.%H:%M:%S.%f'), '+\t\t/data/test/hello')])
    ret = {}
    ret['retcode'] = 0
    ret['stdout'] = '\n'.join(['1517063879.144517494\tM\t\t/data/test/', '1517063875.296592355\t+\t\t/data/test/world', '1517063879.274438467\t+\t\t/data/test/hello'])
    ret['stderr'] = ''
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.diff('myzpool/data@yesterday', 'myzpool/data', parsable=False)

@pytest.mark.slow_test
def test_rollback_success(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs rollback success\n    '
    res = OrderedDict([('rolledback', True)])
    ret = {'pid': 56502, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.rollback('myzpool/mydataset@yesterday')

@pytest.mark.slow_test
def test_rollback_failure(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests zfs rollback failure\n    '
    res = OrderedDict([('rolledback', False), ('error', '\n'.join(["cannot rollback to 'myzpool/mydataset@yesterday': more recent snapshots or bookmarks exist", "use 'recursive=True' to force deletion of the following snapshots and bookmarks:", 'myzpool/mydataset@today']))])
    ret = {'pid': 57471, 'retcode': 1, 'stderr': "cannot rollback to 'myzpool/mydataset@yesterday': more recent snapshots or bookmarks exist\nuse '-r' to force deletion of the following snapshots and bookmarks:\nmyzpool/mydataset@today", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.rollback('myzpool/mydataset@yesterday')

def test_clone_success(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs clone success\n    '
    res = OrderedDict([('cloned', True)])
    ret = {'pid': 64532, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.clone('myzpool/mydataset@yesterday', 'myzpool/yesterday')

def test_clone_failure(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs clone failure\n    '
    res = OrderedDict([('cloned', False), ('error', "cannot create 'myzpool/archive/yesterday': parent does not exist")])
    ret = {'pid': 64864, 'retcode': 1, 'stderr': "cannot create 'myzpool/archive/yesterday': parent does not exist", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.clone('myzpool/mydataset@yesterday', 'myzpool/archive/yesterday')

@pytest.mark.slow_test
def test_promote_success(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs promote success\n    '
    res = OrderedDict([('promoted', True)])
    ret = {'pid': 69075, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.promote('myzpool/yesterday')

def test_promote_failure(utils_patch):
    if False:
        return 10
    '\n    Tests zfs promote failure\n    '
    res = OrderedDict([('promoted', False), ('error', "cannot promote 'myzpool/yesterday': not a cloned filesystem")])
    ret = {'pid': 69209, 'retcode': 1, 'stderr': "cannot promote 'myzpool/yesterday': not a cloned filesystem", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.promote('myzpool/yesterday')

@pytest.mark.slow_test
def test_bookmark_success(utils_patch):
    if False:
        return 10
    '\n    Tests zfs bookmark success\n    '
    with patch('salt.utils.path.which', MagicMock(return_value='/usr/bin/man')):
        res = OrderedDict([('bookmarked', True)])
        ret = {'pid': 20990, 'retcode': 0, 'stderr': '', 'stdout': ''}
        mock_cmd = MagicMock(return_value=ret)
        with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
            assert res == zfs.bookmark('myzpool/mydataset@yesterday', 'myzpool/mydataset#important')

@pytest.mark.slow_test
def test_holds_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs holds success\n    '
    res = OrderedDict([('important', 'Wed Dec 23 21:06 2015'), ('release-1.0', 'Wed Dec 23 21:08 2015')])
    ret = {'pid': 40216, 'retcode': 0, 'stderr': '', 'stdout': 'myzpool/mydataset@baseline\timportant  \tWed Dec 23 21:06 2015\nmyzpool/mydataset@baseline\trelease-1.0\tWed Dec 23 21:08 2015'}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.holds('myzpool/mydataset@baseline')

def test_holds_failure(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs holds failure\n    '
    res = OrderedDict([('error', "cannot open 'myzpool/mydataset@baseline': dataset does not exist")])
    ret = {'pid': 40993, 'retcode': 1, 'stderr': "cannot open 'myzpool/mydataset@baseline': dataset does not exist", 'stdout': 'no datasets available'}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.holds('myzpool/mydataset@baseline')

def test_hold_success(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs hold success\n    '
    res = OrderedDict([('held', True)])
    ret = {'pid': 50876, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.hold('important', 'myzpool/mydataset@baseline', 'myzpool/mydataset@release-1.0')

def test_hold_failure(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs hold failure\n    '
    res = OrderedDict([('held', False), ('error', "cannot hold snapshot 'myzpool/mydataset@baseline': tag already exists on this dataset")])
    ret = {'pid': 51006, 'retcode': 1, 'stderr': "cannot hold snapshot 'myzpool/mydataset@baseline': tag already exists on this dataset", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.hold('important', 'myzpool/mydataset@baseline')

def test_release_success(utils_patch):
    if False:
        return 10
    '\n    Tests zfs release success\n    '
    res = OrderedDict([('released', True)])
    ret = {'pid': 50876, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.release('important', 'myzpool/mydataset@baseline', 'myzpool/mydataset@release-1.0')

@pytest.mark.slow_test
def test_release_failure(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs release failure\n    '
    res = OrderedDict([('released', False), ('error', "cannot release hold from snapshot 'myzpool/mydataset@baseline': no such tag on this dataset")])
    ret = {'pid': 51006, 'retcode': 1, 'stderr': "cannot release hold from snapshot 'myzpool/mydataset@baseline': no such tag on this dataset", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.release('important', 'myzpool/mydataset@baseline')

def test_snapshot_success(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests zfs snapshot success\n    '
    res = OrderedDict([('snapshotted', True)])
    ret = {'pid': 69125, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.snapshot('myzpool/mydataset@baseline')

def test_snapshot_failure(utils_patch):
    if False:
        i = 10
        return i + 15
    '\n    Tests zfs snapshot failure\n    '
    res = OrderedDict([('snapshotted', False), ('error', "cannot create snapshot 'myzpool/mydataset@baseline': dataset already exists")])
    ret = {'pid': 68526, 'retcode': 1, 'stderr': "cannot create snapshot 'myzpool/mydataset@baseline': dataset already exists", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.snapshot('myzpool/mydataset@baseline')

def test_snapshot_failure2(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs snapshot failure\n    '
    res = OrderedDict([('snapshotted', False), ('error', "cannot open 'myzpool/mydataset': dataset does not exist")])
    ret = {'pid': 69256, 'retcode': 2, 'stderr': "cannot open 'myzpool/mydataset': dataset does not exist\nusage:\n\tsnapshot [-r] [-o property=value] ... <filesystem|volume>@<snap> ...\n\nFor the property list, run: zfs set|get\n\nFor the delegated permission list, run: zfs allow|unallow", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.snapshot('myzpool/mydataset@baseline')

def test_set_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs set success\n    '
    res = OrderedDict([('set', True)])
    ret = {'pid': 79736, 'retcode': 0, 'stderr': '', 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.set('myzpool/mydataset', compression='lz4')

@pytest.mark.slow_test
def test_set_failure(utils_patch):
    if False:
        print('Hello World!')
    '\n    Tests zfs set failure\n    '
    res = OrderedDict([('set', False), ('error', "cannot set property for 'myzpool/mydataset': 'canmount' must be one of 'on | off | noauto'")])
    ret = {'pid': 79887, 'retcode': 1, 'stderr': "cannot set property for 'myzpool/mydataset': 'canmount' must be one of 'on | off | noauto'", 'stdout': ''}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.set('myzpool/mydataset', canmount='lz4')

def test_get_success(utils_patch):
    if False:
        while True:
            i = 10
    '\n    Tests zfs get success\n    '
    res = OrderedDict([('myzpool', OrderedDict([('used', OrderedDict([('value', 906238099456)]))]))])
    ret = {'pid': 562, 'retcode': 0, 'stderr': '', 'stdout': 'myzpool\tused\t906238099456'}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.get('myzpool', properties='used', fields='value')

def test_get_parsable_success(utils_patch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests zfs get with parsable output\n    '
    res = OrderedDict([('myzpool', OrderedDict([('used', OrderedDict([('value', '844G')]))]))])
    ret = {'pid': 562, 'retcode': 0, 'stderr': '', 'stdout': 'myzpool\tused\t906238099456'}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(zfs.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(zfs.__utils__, utils_patch):
        assert res == zfs.get('myzpool', properties='used', fields='value', parsable=False)