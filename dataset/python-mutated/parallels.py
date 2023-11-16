"""
Manage Parallels Desktop VMs with ``prlctl`` and ``prlsrvctl``.  Only some of
the prlctl commands implemented so far.  Of those that have been implemented,
not all of the options may have been provided yet.  For a complete reference,
see the `Parallels Desktop Reference Guide
<http://download.parallels.com/desktop/v9/ga/docs/en_US/Parallels%20Command%20Line%20Reference%20Guide.pdf>`_.

This module requires the prlctl binary to be installed to run most functions.
To run parallels.prlsrvctl, the prlsrvctl binary is required.

What has not been implemented yet can be accessed through ``parallels.prlctl``
and ``parallels.prlsrvctl`` (note the preceding double dash ``--`` as
necessary):

.. code-block:: bash

    salt '*' parallels.prlctl installtools macvm runas=macdev
    salt -- '*' parallels.prlctl capture 'macvm --file macvm.display.png' runas=macdev
    salt -- '*' parallels.prlsrvctl set '--mem-limit auto' runas=macdev

.. versionadded:: 2016.3.0
"""
import logging
import re
import shlex
import salt.utils.data
import salt.utils.path
import salt.utils.yaml
from salt.exceptions import CommandExecutionError, SaltInvocationError
__virtualname__ = 'parallels'
__func_alias__ = {'exec_': 'exec'}
log = logging.getLogger(__name__)
GUID_REGEX = re.compile('{?([0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})}?', re.I)

def _normalize_args(args):
    if False:
        return 10
    '\n    Return args as a list of strings\n    '
    if isinstance(args, str):
        return shlex.split(args)
    if isinstance(args, (tuple, list)):
        return [str(arg) for arg in args]
    else:
        return [str(args)]

def _find_guids(guid_string):
    if False:
        return 10
    '\n    Return the set of GUIDs found in guid_string\n\n    :param str guid_string:\n        String containing zero or more GUIDs.  Each GUID may or may not be\n        enclosed in {}\n\n    Example data (this string contains two distinct GUIDs):\n\n    PARENT_SNAPSHOT_ID                      SNAPSHOT_ID\n                                            {a5b8999f-5d95-4aff-82de-e515b0101b66}\n    {a5b8999f-5d95-4aff-82de-e515b0101b66} *{a7345be5-ab66-478c-946e-a6c2caf14909}\n    '
    guids = []
    for found_guid in re.finditer(GUID_REGEX, guid_string):
        if found_guid.groups():
            guids.append(found_guid.group(0).strip('{}'))
    return sorted(list(set(guids)))

def prlsrvctl(sub_cmd, args=None, runas=None):
    if False:
        print('Hello World!')
    "\n    Execute a prlsrvctl command\n\n    .. versionadded:: 2016.11.0\n\n    :param str sub_cmd:\n        prlsrvctl subcommand to execute\n\n    :param str args:\n        The arguments supplied to ``prlsrvctl <sub_cmd>``\n\n    :param str runas:\n        The user that the prlsrvctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.prlsrvctl info runas=macdev\n        salt '*' parallels.prlsrvctl usb list runas=macdev\n        salt -- '*' parallels.prlsrvctl set '--mem-limit auto' runas=macdev\n    "
    if not salt.utils.path.which('prlsrvctl'):
        raise CommandExecutionError('prlsrvctl utility not available')
    cmd = ['prlsrvctl', sub_cmd]
    if args:
        cmd.extend(_normalize_args(args))
    return __salt__['cmd.run'](cmd, runas=runas)

def prlctl(sub_cmd, args=None, runas=None):
    if False:
        print('Hello World!')
    "\n    Execute a prlctl command\n\n    :param str sub_cmd:\n        prlctl subcommand to execute\n\n    :param str args:\n        The arguments supplied to ``prlctl <sub_cmd>``\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.prlctl user list runas=macdev\n        salt '*' parallels.prlctl exec 'macvm uname' runas=macdev\n        salt -- '*' parallels.prlctl capture 'macvm --file macvm.display.png' runas=macdev\n    "
    if not salt.utils.path.which('prlctl'):
        raise CommandExecutionError('prlctl utility not available')
    cmd = ['prlctl', sub_cmd]
    if args:
        cmd.extend(_normalize_args(args))
    return __salt__['cmd.run'](cmd, runas=runas)

def list_vms(name=None, info=False, all=False, args=None, runas=None, template=False):
    if False:
        while True:
            i = 10
    "\n    List information about the VMs\n\n    :param str name:\n        Name/ID of VM to list\n\n        .. versionchanged:: 2016.11.0\n\n            No longer implies ``info=True``\n\n    :param str info:\n        List extra information\n\n    :param bool all:\n        List all non-template VMs\n\n    :param tuple args:\n        Additional arguments given to ``prctl list``\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    :param bool template:\n        List the available virtual machine templates.  The real virtual\n        machines will not be included in the output\n\n        .. versionadded:: 2016.11.0\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.list_vms runas=macdev\n        salt '*' parallels.list_vms name=macvm info=True runas=macdev\n        salt '*' parallels.list_vms info=True runas=macdev\n        salt '*' parallels.list_vms ' -o uuid,status' all=True runas=macdev\n    "
    if args is None:
        args = []
    else:
        args = _normalize_args(args)
    if name:
        args.extend([name])
    if info:
        args.append('--info')
    if all:
        args.append('--all')
    if template:
        args.append('--template')
    return prlctl('list', args, runas=runas)

def clone(name, new_name, linked=False, template=False, runas=None):
    if False:
        print('Hello World!')
    "\n    Clone a VM\n\n    .. versionadded:: 2016.11.0\n\n    :param str name:\n        Name/ID of VM to clone\n\n    :param str new_name:\n        Name of the new VM\n\n    :param bool linked:\n        Create a linked virtual machine.\n\n    :param bool template:\n        Create a virtual machine template instead of a real virtual machine.\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.clone macvm macvm_new runas=macdev\n        salt '*' parallels.clone macvm macvm_templ template=True runas=macdev\n    "
    args = [salt.utils.data.decode(name), '--name', salt.utils.data.decode(new_name)]
    if linked:
        args.append('--linked')
    if template:
        args.append('--template')
    return prlctl('clone', args, runas=runas)

def delete(name, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Delete a VM\n\n    .. versionadded:: 2016.11.0\n\n    :param str name:\n        Name/ID of VM to clone\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.exec macvm 'find /etc/paths.d' runas=macdev\n    "
    return prlctl('delete', salt.utils.data.decode(name), runas=runas)

def exists(name, runas=None):
    if False:
        print('Hello World!')
    "\n    Query whether a VM exists\n\n    .. versionadded:: 2016.11.0\n\n    :param str name:\n        Name/ID of VM\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.exists macvm runas=macdev\n    "
    vm_info = list_vms(name, info=True, runas=runas).splitlines()
    for info_line in vm_info:
        if 'Name: {}'.format(name) in info_line:
            return True
    return False

def start(name, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Start a VM\n\n    :param str name:\n        Name/ID of VM to start\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.start macvm runas=macdev\n    "
    return prlctl('start', salt.utils.data.decode(name), runas=runas)

def stop(name, kill=False, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Stop a VM\n\n    :param str name:\n        Name/ID of VM to stop\n\n    :param bool kill:\n        Perform a hard shutdown\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.stop macvm runas=macdev\n        salt '*' parallels.stop macvm kill=True runas=macdev\n    "
    args = [salt.utils.data.decode(name)]
    if kill:
        args.append('--kill')
    return prlctl('stop', args, runas=runas)

def restart(name, runas=None):
    if False:
        return 10
    "\n    Restart a VM by gracefully shutting it down and then restarting\n    it\n\n    :param str name:\n        Name/ID of VM to restart\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.restart macvm runas=macdev\n    "
    return prlctl('restart', salt.utils.data.decode(name), runas=runas)

def reset(name, runas=None):
    if False:
        print('Hello World!')
    "\n    Reset a VM by performing a hard shutdown and then a restart\n\n    :param str name:\n        Name/ID of VM to reset\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.reset macvm runas=macdev\n    "
    return prlctl('reset', salt.utils.data.decode(name), runas=runas)

def status(name, runas=None):
    if False:
        return 10
    "\n    Status of a VM\n\n    :param str name:\n        Name/ID of VM whose status will be returned\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.status macvm runas=macdev\n    "
    return prlctl('status', salt.utils.data.decode(name), runas=runas)

def exec_(name, command, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run a command on a VM\n\n    :param str name:\n        Name/ID of VM whose exec will be returned\n\n    :param str command:\n        Command to run on the VM\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.exec macvm 'find /etc/paths.d' runas=macdev\n    "
    args = [salt.utils.data.decode(name)]
    args.extend(_normalize_args(command))
    return prlctl('exec', args, runas=runas)

def snapshot_id_to_name(name, snap_id, strict=False, runas=None):
    if False:
        return 10
    "\n    Attempt to convert a snapshot ID to a snapshot name.  If the snapshot has\n    no name or if the ID is not found or invalid, an empty string will be returned\n\n    :param str name:\n        Name/ID of VM whose snapshots are inspected\n\n    :param str snap_id:\n        ID of the snapshot\n\n    :param bool strict:\n        Raise an exception if a name cannot be found for the given ``snap_id``\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example data\n\n    .. code-block:: yaml\n\n        ID: {a5b8999f-5d95-4aff-82de-e515b0101b66}\n        Name: original\n        Date: 2016-03-04 10:50:34\n        Current: yes\n        State: poweroff\n        Description: original state\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.snapshot_id_to_name macvm a5b8999f-5d95-4aff-82de-e515b0101b66 runas=macdev\n    "
    name = salt.utils.data.decode(name)
    if not re.match(GUID_REGEX, snap_id):
        raise SaltInvocationError('Snapshot ID "{}" is not a GUID'.format(salt.utils.data.decode(snap_id)))
    info = prlctl('snapshot-list', [name, '--id', snap_id], runas=runas)
    if not info:
        raise SaltInvocationError('No snapshots for VM "{}" have ID "{}"'.format(name, snap_id))
    try:
        data = salt.utils.yaml.safe_load(info)
    except salt.utils.yaml.YAMLError as err:
        log.warning('Could not interpret snapshot data returned from prlctl: %s', err)
        data = {}
    if isinstance(data, dict):
        snap_name = data.get('Name', '')
        if snap_name is None:
            snap_name = ''
    else:
        log.warning('Could not interpret snapshot data returned from prlctl: data is not formed as a dictionary: %s', data)
        snap_name = ''
    if not snap_name and strict:
        raise SaltInvocationError('Could not find a snapshot name for snapshot ID "{}" of VM "{}"'.format(snap_id, name))
    return salt.utils.data.decode(snap_name)

def snapshot_name_to_id(name, snap_name, strict=False, runas=None):
    if False:
        while True:
            i = 10
    "\n    Attempt to convert a snapshot name to a snapshot ID.  If the name is not\n    found an empty string is returned.  If multiple snapshots share the same\n    name, a list will be returned\n\n    :param str name:\n        Name/ID of VM whose snapshots are inspected\n\n    :param str snap_name:\n        Name of the snapshot\n\n    :param bool strict:\n        Raise an exception if multiple snapshot IDs are found\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.snapshot_id_to_name macvm original runas=macdev\n    "
    name = salt.utils.data.decode(name)
    snap_name = salt.utils.data.decode(snap_name)
    info = prlctl('snapshot-list', name, runas=runas)
    snap_ids = _find_guids(info)
    named_ids = []
    for snap_id in snap_ids:
        if snapshot_id_to_name(name, snap_id, runas=runas) == snap_name:
            named_ids.append(snap_id)
    if not named_ids:
        raise SaltInvocationError('No snapshots for VM "{}" have name "{}"'.format(name, snap_name))
    elif len(named_ids) == 1:
        return named_ids[0]
    else:
        multi_msg = 'Multiple snapshots for VM "{}" have name "{}"'.format(name, snap_name)
        if strict:
            raise SaltInvocationError(multi_msg)
        else:
            log.warning(multi_msg)
        return named_ids

def _validate_snap_name(name, snap_name, strict=True, runas=None):
    if False:
        while True:
            i = 10
    '\n    Validate snapshot name and convert to snapshot ID\n\n    :param str name:\n        Name/ID of VM whose snapshot name is being validated\n\n    :param str snap_name:\n        Name/ID of snapshot\n\n    :param bool strict:\n        Raise an exception if multiple snapshot IDs are found\n\n    :param str runas:\n        The user that the prlctl command will be run as\n    '
    snap_name = salt.utils.data.decode(snap_name)
    if re.match(GUID_REGEX, snap_name):
        return snap_name.strip('{}')
    else:
        return snapshot_name_to_id(name, snap_name, strict=strict, runas=runas)

def list_snapshots(name, snap_name=None, tree=False, names=False, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    List the snapshots\n\n    :param str name:\n        Name/ID of VM whose snapshots will be listed\n\n    :param str snap_id:\n        Name/ID of snapshot to display information about.  If ``tree=True`` is\n        also specified, display the snapshot subtree having this snapshot as\n        the root snapshot\n\n    :param bool tree:\n        List snapshots in tree format rather than tabular format\n\n    :param bool names:\n        List snapshots as ID, name pairs\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.list_snapshots macvm runas=macdev\n        salt '*' parallels.list_snapshots macvm tree=True runas=macdev\n        salt '*' parallels.list_snapshots macvm snap_name=original runas=macdev\n        salt '*' parallels.list_snapshots macvm names=True runas=macdev\n    "
    name = salt.utils.data.decode(name)
    if snap_name:
        snap_name = _validate_snap_name(name, snap_name, runas=runas)
    args = [name]
    if tree:
        args.append('--tree')
    if snap_name:
        args.extend(['--id', snap_name])
    res = prlctl('snapshot-list', args, runas=runas)
    if names:
        snap_ids = _find_guids(res)
        ret = '{:<38}  {}\n'.format('Snapshot ID', 'Snapshot Name')
        for snap_id in snap_ids:
            snap_name = snapshot_id_to_name(name, snap_id, runas=runas)
            ret += '{{{0}}}  {1}\n'.format(snap_id, salt.utils.data.decode(snap_name))
        return ret
    else:
        return res

def snapshot(name, snap_name=None, desc=None, runas=None):
    if False:
        print('Hello World!')
    "\n    Create a snapshot\n\n    :param str name:\n        Name/ID of VM to take a snapshot of\n\n    :param str snap_name:\n        Name of snapshot\n\n    :param str desc:\n        Description of snapshot\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.create_snapshot macvm snap_name=macvm-original runas=macdev\n        salt '*' parallels.create_snapshot macvm snap_name=macvm-updates desc='clean install with updates' runas=macdev\n    "
    name = salt.utils.data.decode(name)
    if snap_name:
        snap_name = salt.utils.data.decode(snap_name)
    args = [name]
    if snap_name:
        args.extend(['--name', snap_name])
    if desc:
        args.extend(['--description', desc])
    return prlctl('snapshot', args, runas=runas)

def delete_snapshot(name, snap_name, runas=None, all=False):
    if False:
        print('Hello World!')
    "\n    Delete a snapshot\n\n    .. note::\n\n        Deleting a snapshot from which other snapshots are dervied will not\n        delete the derived snapshots\n\n    :param str name:\n        Name/ID of VM whose snapshot will be deleted\n\n    :param str snap_name:\n        Name/ID of snapshot to delete\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    :param bool all:\n        Delete all snapshots having the name given\n\n        .. versionadded:: 2016.11.0\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.delete_snapshot macvm 'unneeded snapshot' runas=macdev\n        salt '*' parallels.delete_snapshot macvm 'Snapshot for linked clone' all=True runas=macdev\n    "
    strict = not all
    name = salt.utils.data.decode(name)
    snap_ids = _validate_snap_name(name, snap_name, strict=strict, runas=runas)
    if isinstance(snap_ids, str):
        snap_ids = [snap_ids]
    ret = {}
    for snap_id in snap_ids:
        snap_id = snap_id.strip('{}')
        args = [name, '--id', snap_id]
        ret[snap_id] = prlctl('snapshot-delete', args, runas=runas)
    ret_keys = list(ret.keys())
    if len(ret_keys) == 1:
        return ret[ret_keys[0]]
    else:
        return ret

def revert_snapshot(name, snap_name, runas=None):
    if False:
        return 10
    "\n    Revert a VM to a snapshot\n\n    :param str name:\n        Name/ID of VM to revert to a snapshot\n\n    :param str snap_name:\n        Name/ID of snapshot to revert to\n\n    :param str runas:\n        The user that the prlctl command will be run as\n\n    Example:\n\n    .. code-block:: bash\n\n        salt '*' parallels.revert_snapshot macvm base-with-updates runas=macdev\n    "
    name = salt.utils.data.decode(name)
    snap_name = _validate_snap_name(name, snap_name, runas=runas)
    args = [name, '--id', snap_name]
    return prlctl('snapshot-switch', args, runas=runas)