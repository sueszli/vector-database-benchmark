"""
Support for VirtualBox using the VBoxManage command

.. versionadded:: 2016.3.0

If the ``vboxdrv`` kernel module is not loaded, this module can automatically
load it by configuring ``autoload_vboxdrv`` in ``/etc/salt/minion``:

.. code-block:: yaml

    autoload_vboxdrv: True

The default for this setting is ``False``.

:depends: virtualbox
"""
import logging
import os.path
import re
import salt.utils.files
import salt.utils.path
from salt.exceptions import CommandExecutionError
LOG = logging.getLogger(__name__)
UUID_RE = re.compile('[^{}]'.format('a-zA-Z0-9._-'))
NAME_RE = re.compile('[^{}]'.format('a-zA-Z0-9._-'))

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load the module if VBoxManage is installed\n    '
    if vboxcmd():
        if __opts__.get('autoload_vboxdrv', False) is True:
            if not __salt__['kmod.is_loaded']('vboxdrv'):
                __salt__['kmod.load']('vboxdrv')
        return True
    return (False, 'The vboxmanaged execution module failed to load: VBoxManage is not installed.')

def vboxcmd():
    if False:
        print('Hello World!')
    "\n    Return the location of the VBoxManage command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.vboxcmd\n    "
    return salt.utils.path.which('VBoxManage')

def list_ostypes():
    if False:
        print('Hello World!')
    "\n    List the available OS Types\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.list_ostypes\n    "
    return list_items('ostypes', True, 'ID')

def list_nodes_min():
    if False:
        while True:
            i = 10
    "\n    Return a list of registered VMs, with minimal information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.list_nodes_min\n    "
    ret = {}
    cmd = '{} list vms'.format(vboxcmd())
    for line in salt.modules.cmdmod.run(cmd).splitlines():
        if not line.strip():
            continue
        comps = line.split()
        name = comps[0].replace('"', '')
        ret[name] = True
    return ret

def list_nodes_full():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of registered VMs, with detailed information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.list_nodes_full\n    "
    return list_items('vms', True, 'Name')

def list_nodes():
    if False:
        return 10
    "\n    Return a list of registered VMs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.list_nodes\n    "
    ret = {}
    nodes = list_nodes_full()
    for node in nodes:
        ret[node] = {'id': nodes[node]['UUID'], 'image': nodes[node]['Guest OS'], 'name': nodes[node]['Name'], 'state': None, 'private_ips': [], 'public_ips': []}
        ret[node]['size'] = '{} RAM, {} CPU'.format(nodes[node]['Memory size'], nodes[node]['Number of CPUs'])
    return ret

def start(name):
    if False:
        while True:
            i = 10
    "\n    Start a VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.start my_vm\n    "
    ret = {}
    cmd = '{} startvm {}'.format(vboxcmd(), name)
    ret = salt.modules.cmdmod.run(cmd).splitlines()
    return ret

def stop(name):
    if False:
        return 10
    "\n    Stop a VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.stop my_vm\n    "
    cmd = '{} controlvm {} poweroff'.format(vboxcmd(), name)
    ret = salt.modules.cmdmod.run(cmd).splitlines()
    return ret

def register(filename):
    if False:
        while True:
            i = 10
    "\n    Register a VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.register my_vm_filename\n    "
    if not os.path.isfile(filename):
        raise CommandExecutionError('The specified filename ({}) does not exist.'.format(filename))
    cmd = '{} registervm {}'.format(vboxcmd(), filename)
    ret = salt.modules.cmdmod.run_all(cmd)
    if ret['retcode'] == 0:
        return True
    return ret['stderr']

def unregister(name, delete=False):
    if False:
        i = 10
        return i + 15
    "\n    Unregister a VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.unregister my_vm_filename\n    "
    nodes = list_nodes_min()
    if name not in nodes:
        raise CommandExecutionError('The specified VM ({}) is not registered.'.format(name))
    cmd = '{} unregistervm {}'.format(vboxcmd(), name)
    if delete is True:
        cmd += ' --delete'
    ret = salt.modules.cmdmod.run_all(cmd)
    if ret['retcode'] == 0:
        return True
    return ret['stderr']

def destroy(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Unregister and destroy a VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vboxmanage.destroy my_vm\n    "
    return unregister(name, True)

def create(name, groups=None, ostype=None, register=True, basefolder=None, new_uuid=None, **kwargs):
    if False:
        return 10
    "\n    Create a new VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'hypervisor' vboxmanage.create <name>\n    "
    nodes = list_nodes_min()
    if name in nodes:
        raise CommandExecutionError('The specified VM ({}) is already registered.'.format(name))
    params = ''
    if name:
        if NAME_RE.search(name):
            raise CommandExecutionError('New VM name contains invalid characters')
        params += ' --name {}'.format(name)
    if groups:
        if isinstance(groups, str):
            groups = [groups]
        if isinstance(groups, list):
            params += ' --groups {}'.format(','.join(groups))
        else:
            raise CommandExecutionError('groups must be either a string or a list of strings')
    ostypes = list_ostypes()
    if ostype not in ostypes:
        raise CommandExecutionError('The specified OS type ({}) is not available.'.format(name))
    else:
        params += ' --ostype ' + ostype
    if register is True:
        params += ' --register'
    if basefolder:
        if not os.path.exists(basefolder):
            raise CommandExecutionError('basefolder {} was not found'.format(basefolder))
        params += ' --basefolder {}'.format(basefolder)
    if new_uuid:
        if NAME_RE.search(new_uuid):
            raise CommandExecutionError('New UUID contains invalid characters')
        params += ' --uuid {}'.format(new_uuid)
    cmd = '{} create {}'.format(vboxcmd(), params)
    ret = salt.modules.cmdmod.run_all(cmd)
    if ret['retcode'] == 0:
        return True
    return ret['stderr']

def clonevm(name=None, uuid=None, new_name=None, snapshot_uuid=None, snapshot_name=None, mode='machine', options=None, basefolder=None, new_uuid=None, register=False, groups=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Clone a new VM from an existing VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'hypervisor' vboxmanage.clonevm <name> <new_name>\n    "
    if name and uuid or (not name and (not uuid)):
        raise CommandExecutionError('Either a name or a uuid must be specified, but not both.')
    params = ''
    nodes_names = list_nodes_min()
    nodes_uuids = list_items('vms', True, 'UUID').keys()
    if name:
        if name not in nodes_names:
            raise CommandExecutionError('The specified VM ({}) is not registered.'.format(name))
        params += ' ' + name
    elif uuid:
        if uuid not in nodes_uuids:
            raise CommandExecutionError('The specified VM ({}) is not registered.'.format(name))
        params += ' ' + uuid
    if snapshot_name and snapshot_uuid:
        raise CommandExecutionError('Either a snapshot_name or a snapshot_uuid may be specified, but not both')
    if snapshot_name:
        if NAME_RE.search(snapshot_name):
            raise CommandExecutionError('Snapshot name contains invalid characters')
        params += ' --snapshot {}'.format(snapshot_name)
    elif snapshot_uuid:
        if UUID_RE.search(snapshot_uuid):
            raise CommandExecutionError('Snapshot name contains invalid characters')
        params += ' --snapshot {}'.format(snapshot_uuid)
    valid_modes = ('machine', 'machineandchildren', 'all')
    if mode and mode not in valid_modes:
        raise CommandExecutionError('Mode must be one of: {} (default "machine")'.format(', '.join(valid_modes)))
    else:
        params += ' --mode ' + mode
    valid_options = ('link', 'keepallmacs', 'keepnatmacs', 'keepdisknames')
    if options and options not in valid_options:
        raise CommandExecutionError('If specified, options must be one of: {}'.format(', '.join(valid_options)))
    else:
        params += ' --options ' + options
    if new_name:
        if NAME_RE.search(new_name):
            raise CommandExecutionError('New name contains invalid characters')
        params += ' --name {}'.format(new_name)
    if groups:
        if isinstance(groups, str):
            groups = [groups]
        if isinstance(groups, list):
            params += ' --groups {}'.format(','.join(groups))
        else:
            raise CommandExecutionError('groups must be either a string or a list of strings')
    if basefolder:
        if not os.path.exists(basefolder):
            raise CommandExecutionError('basefolder {} was not found'.format(basefolder))
        params += ' --basefolder {}'.format(basefolder)
    if new_uuid:
        if NAME_RE.search(new_uuid):
            raise CommandExecutionError('New UUID contains invalid characters')
        params += ' --uuid {}'.format(new_uuid)
    if register is True:
        params += ' --register'
    cmd = '{} clonevm {}'.format(vboxcmd(), name)
    ret = salt.modules.cmdmod.run_all(cmd)
    if ret['retcode'] == 0:
        return True
    return ret['stderr']

def clonemedium(medium, uuid_in=None, file_in=None, uuid_out=None, file_out=None, mformat=None, variant=None, existing=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Clone a new VM from an existing VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'hypervisor' vboxmanage.clonemedium <name> <new_name>\n    "
    params = ''
    valid_mediums = ('disk', 'dvd', 'floppy')
    if medium in valid_mediums:
        params += medium
    else:
        raise CommandExecutionError('Medium must be one of: {}.'.format(', '.join(valid_mediums)))
    if uuid_in and file_in or (not uuid_in and (not file_in)):
        raise CommandExecutionError('Either uuid_in or file_in must be used, but not both.')
    if uuid_in:
        if medium == 'disk':
            item = 'hdds'
        elif medium == 'dvd':
            item = 'dvds'
        elif medium == 'floppy':
            item = 'floppies'
        items = list_items(item)
        if uuid_in not in items:
            raise CommandExecutionError('UUID {} was not found'.format(uuid_in))
        params += ' ' + uuid_in
    elif file_in:
        if not os.path.exists(file_in):
            raise CommandExecutionError('File {} was not found'.format(file_in))
        params += ' ' + file_in
    if uuid_out and file_out or (not uuid_out and (not file_out)):
        raise CommandExecutionError('Either uuid_out or file_out must be used, but not both.')
    if uuid_out:
        params += ' ' + uuid_out
    elif file_out:
        try:
            salt.utils.files.fopen(file_out, 'w').close()
            os.unlink(file_out)
            params += ' ' + file_out
        except OSError:
            raise CommandExecutionError('{} is not a valid filename'.format(file_out))
    if mformat:
        valid_mformat = ('VDI', 'VMDK', 'VHD', 'RAW')
        if mformat not in valid_mformat:
            raise CommandExecutionError('If specified, mformat must be one of: {}'.format(', '.join(valid_mformat)))
        else:
            params += ' --format ' + mformat
    valid_variant = ('Standard', 'Fixed', 'Split2G', 'Stream', 'ESX')
    if variant and variant not in valid_variant:
        if not os.path.exists(file_in):
            raise CommandExecutionError('If specified, variant must be one of: {}'.format(', '.join(valid_variant)))
        else:
            params += ' --variant ' + variant
    if existing:
        params += ' --existing'
    cmd = '{} clonemedium {}'.format(vboxcmd(), params)
    ret = salt.modules.cmdmod.run_all(cmd)
    if ret['retcode'] == 0:
        return True
    return ret['stderr']

def list_items(item, details=False, group_by='UUID'):
    if False:
        print('Hello World!')
    "\n    Return a list of a specific type of item. The following items are available:\n\n        vms\n        runningvms\n        ostypes\n        hostdvds\n        hostfloppies\n        intnets\n        bridgedifs\n        hostonlyifs\n        natnets\n        dhcpservers\n        hostinfo\n        hostcpuids\n        hddbackends\n        hdds\n        dvds\n        floppies\n        usbhost\n        usbfilters\n        systemproperties\n        extpacks\n        groups\n        webcams\n        screenshotformats\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'hypervisor' vboxmanage.items <item>\n        salt 'hypervisor' vboxmanage.items <item> details=True\n        salt 'hypervisor' vboxmanage.items <item> details=True group_by=Name\n\n    Some items do not display well, or at all, unless ``details`` is set to\n    ``True``. By default, items are grouped by the ``UUID`` field, but not all\n    items contain that field. In those cases, another field must be specified.\n    "
    types = ('vms', 'runningvms', 'ostypes', 'hostdvds', 'hostfloppies', 'intnets', 'bridgedifs', 'hostonlyifs', 'natnets', 'dhcpservers', 'hostinfo', 'hostcpuids', 'hddbackends', 'hdds', 'dvds', 'floppies', 'usbhost', 'usbfilters', 'systemproperties', 'extpacks', 'groups', 'webcams', 'screenshotformats')
    if item not in types:
        raise CommandExecutionError('Item must be one of: {}.'.format(', '.join(types)))
    flag = ''
    if details is True:
        flag = ' -l'
    ret = {}
    tmp_id = None
    tmp_dict = {}
    cmd = '{} list{} {}'.format(vboxcmd(), flag, item)
    for line in salt.modules.cmdmod.run(cmd).splitlines():
        if not line.strip():
            continue
        comps = line.split(':')
        if not comps:
            continue
        if tmp_id is not None:
            ret[tmp_id] = tmp_dict
        line_val = ':'.join(comps[1:]).strip()
        if comps[0] == group_by:
            tmp_id = line_val
            tmp_dict = {}
        tmp_dict[comps[0]] = line_val
    return ret