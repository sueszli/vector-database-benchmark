"""

Management of Pure Storage FlashBlade

Installation Prerequisites
--------------------------
- You will need the ``purity_fb`` python package in your python installation
  path that is running salt.

  .. code-block:: bash

      pip install purity_fb

- Configure Pure Storage FlashBlade authentication. Use one of the following
  three methods.

  1) From the minion config

  .. code-block:: yaml

        pure_tags:
          fb:
            san_ip: management vip or hostname for the FlashBlade
            api_token: A valid api token for the FlashBlade being managed

  2) From environment (PUREFB_IP and PUREFB_API)
  3) From the pillar (PUREFB_IP and PUREFB_API)

:maintainer: Simon Dodsley (simon@purestorage.com)
:maturity: new
:requires: purity_fb
:platform: all

.. versionadded:: 2019.2.0

"""
import os
from datetime import datetime
from salt.exceptions import CommandExecutionError
try:
    from purity_fb import FileSystem, FileSystemSnapshot, NfsRule, ProtocolRule, PurityFb, SnapshotSuffix, rest
    HAS_PURITY_FB = True
except ImportError:
    HAS_PURITY_FB = False
__docformat__ = 'restructuredtext en'
__virtualname__ = 'purefb'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine whether or not to load this module\n    '
    if HAS_PURITY_FB:
        return __virtualname__
    return (False, 'purefb execution module not loaded: purity_fb python library not available.')

def _get_blade():
    if False:
        while True:
            i = 10
    '\n    Get Pure Storage FlasBlade configuration\n\n    1) From the minion config\n        pure_tags:\n          fb:\n            san_ip: management vip or hostname for the FlashBlade\n            api_token: A valid api token for the FlashBlade being managed\n    2) From environment (PUREFB_IP and PUREFB_API)\n    3) From the pillar (PUREFB_IP and PUREFB_API)\n\n    '
    try:
        blade_name = __opts__['pure_tags']['fb'].get('san_ip')
        api_token = __opts__['pure_tags']['fb'].get('api_token')
        if blade_name and api:
            blade = PurityFb(blade_name)
            blade.disable_verify_ssl()
    except (KeyError, NameError, TypeError):
        try:
            blade_name = os.environ.get('PUREFB_IP')
            api_token = os.environ.get('PUREFB_API')
            if blade_name:
                blade = PurityFb(blade_name)
                blade.disable_verify_ssl()
        except (ValueError, KeyError, NameError):
            try:
                api_token = __pillar__['PUREFB_API']
                blade = PurityFb(__pillar__['PUREFB_IP'])
                blade.disable_verify_ssl()
            except (KeyError, NameError):
                raise CommandExecutionError('No Pure Storage FlashBlade credentials found.')
    try:
        blade.login(api_token)
    except Exception:
        raise CommandExecutionError('Pure Storage FlashBlade authentication failed.')
    return blade

def _get_fs(name, blade):
    if False:
        i = 10
        return i + 15
    '\n    Private function to\n    check for existence of a filesystem\n    '
    _fs = []
    _fs.append(name)
    try:
        res = blade.file_systems.list_file_systems(names=_fs)
        return res.items[0]
    except rest.ApiException:
        return None

def _get_snapshot(name, suffix, blade):
    if False:
        i = 10
        return i + 15
    '\n    Return name of Snapshot\n    or None\n    '
    try:
        filt = "source='{}' and suffix='{}'".format(name, suffix)
        res = blade.file_system_snapshots.list_file_system_snapshots(filter=filt)
        return res.items[0]
    except rest.ApiException:
        return None

def _get_deleted_fs(name, blade):
    if False:
        for i in range(10):
            print('nop')
    '\n    Private function to check\n    if a file systeem has already been deleted\n    '
    try:
        _fs = _get_fs(name, blade)
        if _fs and _fs.destroyed:
            return _fs
    except rest.ApiException:
        return None

def snap_create(name, suffix=None):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Create a filesystem snapshot on a Pure Storage FlashBlade.\n\n    Will return False if filesystem selected to snap does not exist.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem to snapshot\n    suffix : string\n        if specificed forces snapshot name suffix. If not specified defaults to timestamp.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.snap_create foo\n        salt '*' purefb.snap_create foo suffix=bar\n\n    "
    blade = _get_blade()
    if suffix is None:
        suffix = 'snap-' + str((datetime.utcnow() - datetime(1970, 1, 1, 0, 0, 0, 0)).total_seconds())
        suffix = suffix.replace('.', '')
    if _get_fs(name, blade) is not None:
        try:
            source = []
            source.append(name)
            blade.file_system_snapshots.create_file_system_snapshots(sources=source, suffix=SnapshotSuffix(suffix))
            return True
        except rest.ApiException:
            return False
    else:
        return False

def snap_delete(name, suffix=None, eradicate=False):
    if False:
        return 10
    "\n\n    Delete a filesystem snapshot on a Pure Storage FlashBlade.\n\n    Will return False if selected snapshot does not exist.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem\n    suffix : string\n        name of snapshot\n    eradicate : boolean\n        Eradicate snapshot after deletion if True. Default is False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.snap_delete foo suffix=snap eradicate=True\n\n    "
    blade = _get_blade()
    if _get_snapshot(name, suffix, blade) is not None:
        try:
            snapname = name + '.' + suffix
            new_attr = FileSystemSnapshot(destroyed=True)
            blade.file_system_snapshots.update_file_system_snapshots(name=snapname, attributes=new_attr)
        except rest.ApiException:
            return False
        if eradicate is True:
            try:
                blade.file_system_snapshots.delete_file_system_snapshots(name=snapname)
                return True
            except rest.ApiException:
                return False
        else:
            return True
    else:
        return False

def snap_eradicate(name, suffix=None):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Eradicate a deleted filesystem snapshot on a Pure Storage FlashBlade.\n\n    Will return False if snapshot is not in a deleted state.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem\n    suffix : string\n        name of snapshot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.snap_eradicate foo suffix=snap\n\n    "
    blade = _get_blade()
    if _get_snapshot(name, suffix, blade) is not None:
        snapname = name + '.' + suffix
        try:
            blade.file_system_snapshots.delete_file_system_snapshots(name=snapname)
            return True
        except rest.ApiException:
            return False
    else:
        return False

def fs_create(name, size=None, proto='NFS', nfs_rules='*(rw,no_root_squash)', snapshot=False):
    if False:
        print('Hello World!')
    "\n\n    Create a filesystem on a Pure Storage FlashBlade.\n\n    Will return False if filesystem already exists.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem (truncated to 63 characters)\n    proto : string\n        (Optional) Sharing protocol (NFS, CIFS or HTTP). If not specified default is NFS\n    snapshot: boolean\n        (Optional) Are snapshots enabled on the filesystem. Default is False\n    nfs_rules : string\n        (Optional) export rules for NFS. If not specified default is\n        ``*(rw,no_root_squash)``. Refer to Pure Storage documentation for\n        formatting rules.\n    size : string\n        if specified capacity of filesystem. If not specified default to 32G.\n        Refer to Pure Storage documentation for formatting rules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.fs_create foo proto=CIFS\n        salt '*' purefb.fs_create foo size=10T\n\n    "
    if len(name) > 63:
        name = name[0:63]
    blade = _get_blade()
    print(proto)
    if _get_fs(name, blade) is None:
        if size is None:
            size = __utils__['stringutils.human_to_bytes']('32G')
        else:
            size = __utils__['stringutils.human_to_bytes'](size)
        if proto.lower() == 'nfs':
            fs_obj = FileSystem(name=name, provisioned=size, fast_remove_directory_enabled=True, snapshot_directory_enabled=snapshot, nfs=NfsRule(enabled=True, rules=nfs_rules))
        elif proto.lower() == 'cifs':
            fs_obj = FileSystem(name=name, provisioned=size, fast_remove_directory_enabled=True, snapshot_directory_enabled=snapshot, smb=ProtocolRule(enabled=True))
        elif proto.lower() == 'http':
            fs_obj = FileSystem(name=name, provisioned=size, fast_remove_directory_enabled=True, snapshot_directory_enabled=snapshot, http=ProtocolRule(enabled=True))
        else:
            return False
        try:
            blade.file_systems.create_file_systems(fs_obj)
            return True
        except rest.ApiException:
            return False
    else:
        return False

def fs_delete(name, eradicate=False):
    if False:
        while True:
            i = 10
    "\n\n    Delete a share on a Pure Storage FlashBlade.\n\n    Will return False if filesystem doesn't exist or is already in a deleted state.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem\n    eradicate : boolean\n        (Optional) Eradicate filesystem after deletion if True. Default is False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.fs_delete foo eradicate=True\n\n    "
    blade = _get_blade()
    if _get_fs(name, blade) is not None:
        try:
            blade.file_systems.update_file_systems(name=name, attributes=FileSystem(nfs=NfsRule(enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True))
        except rest.ApiException:
            return False
        if eradicate is True:
            try:
                blade.file_systems.delete_file_systems(name)
                return True
            except rest.ApiException:
                return False
        else:
            return True
    else:
        return False

def fs_eradicate(name):
    if False:
        return 10
    "\n\n    Eradicate a deleted filesystem on a Pure Storage FlashBlade.\n\n    Will return False is filesystem is not in a deleted state.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.fs_eradicate foo\n\n    "
    blade = _get_blade()
    if _get_deleted_fs(name, blade) is not None:
        try:
            blade.file_systems.delete_file_systems(name)
            return True
        except rest.ApiException:
            return False
    else:
        return False

def fs_extend(name, size):
    if False:
        i = 10
        return i + 15
    "\n\n    Resize an existing filesystem on a Pure Storage FlashBlade.\n\n    Will return False if new size is less than or equal to existing size.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem\n    size : string\n        New capacity of filesystem.\n        Refer to Pure Storage documentation for formatting rules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.fs_extend foo 10T\n\n    "
    attr = {}
    blade = _get_blade()
    _fs = _get_fs(name, blade)
    if _fs is not None:
        if __utils__['stringutils.human_to_bytes'](size) > _fs.provisioned:
            try:
                attr['provisioned'] = __utils__['stringutils.human_to_bytes'](size)
                n_attr = FileSystem(**attr)
                blade.file_systems.update_file_systems(name=name, attributes=n_attr)
                return True
            except rest.ApiException:
                return False
        else:
            return False
    else:
        return False

def fs_update(name, rules, snapshot=False):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Update filesystem on a Pure Storage FlashBlade.\n\n    Allows for change of NFS export rules and enabling/disabled\n    of snapshotting capability.\n\n    .. versionadded:: 2019.2.0\n\n    name : string\n        name of filesystem\n    rules : string\n        NFS export rules for filesystem\n        Refer to Pure Storage documentation for formatting rules.\n    snapshot: boolean\n        (Optional) Enable/Disable snapshots on the filesystem. Default is False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefb.fs_nfs_update foo rules='10.234.112.23(ro), 10.234.112.24(rw)' snapshot=True\n\n    "
    blade = _get_blade()
    attr = {}
    _fs = _get_fs(name, blade)
    if _fs is not None:
        try:
            if _fs.nfs.enabled:
                attr['nfs'] = NfsRule(rules=rules)
            attr['snapshot_directory_enabled'] = snapshot
            n_attr = FileSystem(**attr)
            blade.file_systems.update_file_systems(name=name, attributes=n_attr)
            return True
        except rest.ApiException:
            return False
    else:
        return False