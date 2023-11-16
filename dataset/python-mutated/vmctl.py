"""
Manage vms running on the OpenBSD VMM hypervisor using vmctl(8).

.. versionadded:: 2019.2.0

:codeauthor: ``Jasper Lievisse Adriaanse <jasper@openbsd.org>``

.. note::

    This module requires the `vmd` service to be running on the OpenBSD
    target machine.
"""
import logging
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only works on OpenBSD with vmctl(8) present.\n    '
    if __grains__['os'] == 'OpenBSD' and salt.utils.path.which('vmctl'):
        return True
    return (False, 'The vmm execution module cannot be loaded: either the system is not OpenBSD or the vmctl binary was not found')

def _id_to_name(id):
    if False:
        return 10
    '\n    Lookup the name associated with a VM id.\n    '
    vm = status(id=id)
    if vm == {}:
        return None
    else:
        return vm['name']

def create_disk(name, size):
    if False:
        return 10
    "\n    Create a VMM disk with the specified `name` and `size`.\n\n    size:\n        Size in megabytes, or use a specifier such as M, G, T.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vmctl.create_disk /path/to/disk.img size=10G\n    "
    ret = False
    cmd = 'vmctl create {} -s {}'.format(name, size)
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret = True
    else:
        raise CommandExecutionError('Problem encountered creating disk image', info={'errors': [result['stderr']], 'changes': ret})
    return ret

def load(path):
    if False:
        i = 10
        return i + 15
    "\n    Load additional configuration from the specified file.\n\n    path\n        Path to the configuration file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vmctl.load path=/etc/vm.switches.conf\n    "
    ret = False
    cmd = 'vmctl load {}'.format(path)
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret = True
    else:
        raise CommandExecutionError('Problem encountered running vmctl', info={'errors': [result['stderr']], 'changes': ret})
    return ret

def reload():
    if False:
        print('Hello World!')
    "\n    Remove all stopped VMs and reload configuration from the default configuration file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vmctl.reload\n    "
    ret = False
    cmd = 'vmctl reload'
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret = True
    else:
        raise CommandExecutionError('Problem encountered running vmctl', info={'errors': [result['stderr']], 'changes': ret})
    return ret

def reset(all=False, vms=False, switches=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reset the running state of VMM or a subsystem.\n\n    all:\n        Reset the running state.\n\n    switches:\n        Reset the configured switches.\n\n    vms:\n        Reset and terminate all VMs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vmctl.reset all=True\n    "
    ret = False
    cmd = ['vmctl', 'reset']
    if all:
        cmd.append('all')
    elif vms:
        cmd.append('vms')
    elif switches:
        cmd.append('switches')
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret = True
    else:
        raise CommandExecutionError('Problem encountered running vmctl', info={'errors': [result['stderr']], 'changes': ret})
    return ret

def start(name=None, id=None, bootpath=None, disk=None, disks=None, local_iface=False, memory=None, nics=0, switch=None):
    if False:
        return 10
    '\n    Starts a VM defined by the specified parameters.\n    When both a name and id are provided, the id is ignored.\n\n    name:\n        Name of the defined VM.\n\n    id:\n        VM id.\n\n    bootpath:\n        Path to a kernel or BIOS image to load.\n\n    disk:\n        Path to a single disk to use.\n\n    disks:\n        List of multiple disks to use.\n\n    local_iface:\n        Whether to add a local network interface. See "LOCAL INTERFACES"\n        in the vmctl(8) manual page for more information.\n\n    memory:\n        Memory size of the VM specified in megabytes.\n\n    switch:\n        Add a network interface that is attached to the specified\n        virtual switch on the host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' vmctl.start 2   # start VM with id 2\n        salt \'*\' vmctl.start name=web1 bootpath=\'/bsd.rd\' nics=2 memory=512M disk=\'/disk.img\'\n    '
    ret = {'changes': False, 'console': None}
    cmd = ['vmctl', 'start']
    if not (name or id):
        raise SaltInvocationError('Must provide either "name" or "id"')
    elif name:
        cmd.append(name)
    else:
        cmd.append(id)
        name = _id_to_name(id)
    if nics > 0:
        cmd.append('-i {}'.format(nics))
    if bootpath:
        cmd.extend(['-b', bootpath])
    if memory:
        cmd.append('-m {}'.format(memory))
    if switch:
        cmd.append('-n {}'.format(switch))
    if local_iface:
        cmd.append('-L')
    if disk and disks:
        raise SaltInvocationError('Must provide either "disks" or "disk"')
    if disk:
        cmd.extend(['-d', disk])
    if disks:
        cmd.extend((['-d', x] for x in disks))
    if len(cmd) > 3:
        vmstate = status(name)
        if vmstate:
            ret['comment'] = 'VM already exists and cannot be redefined'
            return ret
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        ret['changes'] = True
        m = re.match('.*successfully, tty (\\/dev.*)', result['stderr'])
        if m:
            ret['console'] = m.groups()[0]
        else:
            m = re.match('.*Operation already in progress$', result['stderr'])
            if m:
                ret['changes'] = False
    else:
        raise CommandExecutionError('Problem encountered running vmctl', info={'errors': [result['stderr']], 'changes': ret})
    return ret

def status(name=None, id=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    List VMs running on the host, or only the VM specified by ``id``.  When\n    both a name and id are provided, the id is ignored.\n\n    name:\n        Name of the defined VM.\n\n    id:\n        VM id.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vmctl.status           # to list all VMs\n        salt '*' vmctl.status name=web1 # to get a single VM\n    "
    ret = {}
    cmd = ['vmctl', 'status']
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered running vmctl', info={'error': [result['stderr']], 'changes': ret})
    header = result['stdout'].splitlines()[0].split()
    header = [x.lower() for x in header]
    for line in result['stdout'].splitlines()[1:]:
        data = line.split()
        vm = dict(list(zip(header, data)))
        vmname = vm.pop('name')
        if vm['pid'] == '-':
            vm['state'] = 'stopped'
        elif vmname and data[-2] == '-':
            vm['state'] = data[-1]
        else:
            vm['state'] = 'running'
        if id and int(vm['id']) == id:
            return {vmname: vm}
        elif name and vmname == name:
            return {vmname: vm}
        else:
            ret[vmname] = vm
    if id or name:
        return {}
    return ret

def stop(name=None, id=None):
    if False:
        while True:
            i = 10
    "\n    Stop (terminate) the VM identified by the given id or name.\n    When both a name and id are provided, the id is ignored.\n\n    name:\n        Name of the defined VM.\n\n    id:\n        VM id.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vmctl.stop name=alpine\n    "
    ret = {}
    cmd = ['vmctl', 'stop']
    if not (name or id):
        raise SaltInvocationError('Must provide either "name" or "id"')
    elif name:
        cmd.append(name)
    else:
        cmd.append(id)
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if result['retcode'] == 0:
        if re.match('^vmctl: sent request to terminate vm.*', result['stderr']):
            ret['changes'] = True
        else:
            ret['changes'] = False
    else:
        raise CommandExecutionError('Problem encountered running vmctl', info={'errors': [result['stderr']], 'changes': ret})
    return ret