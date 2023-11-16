"""
Work with virtual machines managed by Vagrant.

.. versionadded:: 2018.3.0

Mapping between a Salt node id and the Vagrant machine name
(and the path to the Vagrantfile where it is defined)
is stored in a Salt sdb database on the Vagrant host (minion) machine.
In order to use this module, sdb must be configured. An SQLite
database is the recommended storage method.  The URI used for
the sdb lookup is "sdb://vagrant_sdb_data".

requirements:
   - the VM host machine must have salt-minion, Vagrant and a vm provider installed.
   - the VM host must have a valid definition for `sdb://vagrant_sdb_data`

    Configuration example:

    .. code-block:: yaml

        # file /etc/salt/minion.d/vagrant_sdb.conf
        vagrant_sdb_data:
          driver: sqlite3
          database: /var/cache/salt/vagrant.sqlite
          table: sdb
          create_table: True

"""
import logging
import os
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
from salt._compat import ipaddress
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'vagrant'
VAGRANT_SDB_URL = 'sdb://vagrant_sdb_data/'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    run Vagrant commands if possible\n    '
    if salt.utils.path.which('vagrant') is None:
        return (False, 'The vagrant module could not be loaded: vagrant command not found')
    return __virtualname__

def _build_sdb_uri(key):
    if False:
        while True:
            i = 10
    '\n    returns string used to fetch data for "key" from the sdb store.\n\n    Salt node id\'s are used as the key for vm_ dicts.\n\n    '
    return f'{VAGRANT_SDB_URL}{key}'

def _build_machine_uri(machine, cwd):
    if False:
        return 10
    "\n    returns string used to fetch id names from the sdb store.\n\n    the cwd and machine name are concatenated with '?' which should\n    never collide with a Salt node id -- which is important since we\n    will be storing both in the same table.\n    "
    key = f'{machine}?{os.path.abspath(cwd)}'
    return _build_sdb_uri(key)

def _update_vm_info(name, vm_):
    if False:
        print('Hello World!')
    'store the vm_ information keyed by name'
    __utils__['sdb.sdb_set'](_build_sdb_uri(name), vm_, __opts__)
    if vm_['machine']:
        __utils__['sdb.sdb_set'](_build_machine_uri(vm_['machine'], vm_.get('cwd', '.')), name, __opts__)

def get_vm_info(name):
    if False:
        return 10
    "\n    get the information for a VM.\n\n    :param name: salt_id name\n    :return: dictionary of {'machine': x, 'cwd': y, ...}.\n    "
    try:
        vm_ = __utils__['sdb.sdb_get'](_build_sdb_uri(name), __opts__)
    except KeyError:
        raise SaltInvocationError('Probable sdb driver not found. Check your configuration.')
    if vm_ is None or 'machine' not in vm_:
        raise SaltInvocationError(f'No Vagrant machine defined for Salt_id {name}')
    return vm_

def get_machine_id(machine, cwd):
    if False:
        i = 10
        return i + 15
    '\n    returns the salt_id name of the Vagrant VM\n\n    :param machine: the Vagrant machine name\n    :param cwd: the path to Vagrantfile\n    :return: salt_id name\n    '
    name = __utils__['sdb.sdb_get'](_build_machine_uri(machine, cwd), __opts__)
    return name

def _erase_vm_info(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    erase the information for a VM the we are destroying.\n\n    some sdb drivers (such as the SQLite driver we expect to use)\n    do not have a `delete` method, so if the delete fails, we have\n    to replace the with a blank entry.\n    '
    try:
        vm_ = get_vm_info(name)
        if vm_['machine']:
            key = _build_machine_uri(vm_['machine'], vm_.get('cwd', '.'))
            try:
                __utils__['sdb.sdb_delete'](key, __opts__)
            except KeyError:
                __utils__['sdb.sdb_set'](key, None, __opts__)
    except Exception:
        pass
    uri = _build_sdb_uri(name)
    try:
        __utils__['sdb.sdb_delete'](uri, __opts__)
    except KeyError:
        __utils__['sdb.sdb_set'](uri, {}, __opts__)
    except Exception:
        pass

def _vagrant_ssh_config(vm_):
    if False:
        i = 10
        return i + 15
    "\n    get the information for ssh communication from the new VM\n\n    :param vm_: the VM's info as we have it now\n    :return: dictionary of ssh stuff\n    "
    machine = vm_['machine']
    log.info('requesting vagrant ssh-config for VM %s', machine or '(default)')
    cmd = f'vagrant ssh-config {machine}'
    reply = __salt__['cmd.shell'](cmd, runas=vm_.get('runas'), cwd=vm_.get('cwd'), ignore_retcode=True)
    ssh_config = {}
    for line in reply.split('\n'):
        tokens = line.strip().split()
        if len(tokens) == 2:
            ssh_config[tokens[0]] = tokens[1]
    log.debug('ssh_config=%s', repr(ssh_config))
    return ssh_config

def version():
    if False:
        return 10
    "\n    Return the version of Vagrant on the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vagrant.version\n    "
    cmd = 'vagrant -v'
    return __salt__['cmd.shell'](cmd)

def list_domains():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of the salt_id names of all available Vagrant VMs on\n    this host without regard to the path where they are defined.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vagrant.list_domains --log-level=info\n\n    The log shows information about all known Vagrant environments\n    on this machine. This data is cached and may not be completely\n    up-to-date.\n    "
    vms = []
    cmd = 'vagrant global-status'
    reply = __salt__['cmd.shell'](cmd)
    log.debug('--->\n%s', reply)
    for line in reply.split('\n'):
        tokens = line.strip().split()
        try:
            _ = int(tokens[0], 16)
        except (ValueError, IndexError):
            continue
        machine = tokens[1]
        cwd = tokens[-1]
        name = get_machine_id(machine, cwd)
        if name:
            vms.append(name)
    return vms

def list_active_vms(cwd=None):
    if False:
        print('Hello World!')
    "\n    Return a list of machine names for active virtual machine on the host,\n    which are defined in the Vagrantfile at the indicated path.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vagrant.list_active_vms  cwd=/projects/project_1\n    "
    vms = []
    cmd = 'vagrant status'
    reply = __salt__['cmd.shell'](cmd, cwd=cwd)
    log.info('--->\n%s', reply)
    for line in reply.split('\n'):
        tokens = line.strip().split()
        if len(tokens) > 1:
            if tokens[1] == 'running':
                vms.append(tokens[0])
    return vms

def list_inactive_vms(cwd=None):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of machine names for inactive virtual machine on the host,\n    which are defined in the Vagrantfile at the indicated path.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.list_inactive_vms cwd=/projects/project_1\n    "
    vms = []
    cmd = 'vagrant status'
    reply = __salt__['cmd.shell'](cmd, cwd=cwd)
    log.info('--->\n%s', reply)
    for line in reply.split('\n'):
        tokens = line.strip().split()
        if len(tokens) > 1 and tokens[-1].endswith(')'):
            if tokens[1] != 'running':
                vms.append(tokens[0])
    return vms

def vm_state(name='', cwd=None):
    if False:
        while True:
            i = 10
    "\n    Return list of information for all the vms indicating their state.\n\n    If you pass a VM name in as an argument then it will return info\n    for just the named VM, otherwise it will return all VMs defined by\n    the Vagrantfile in the `cwd` directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vagrant.vm_state <name>  cwd=/projects/project_1\n\n    returns a list of dictionaries with machine name, state, provider,\n    and salt_id name.\n\n    .. code-block:: python\n\n        datum = {'machine': _, # Vagrant machine name,\n                 'state': _, # string indicating machine state, like 'running'\n                 'provider': _, # the Vagrant VM provider\n                 'name': _} # salt_id name\n\n    Known bug: if there are multiple machines in your Vagrantfile, and you request\n    the status of the ``primary`` machine, which you defined by leaving the ``machine``\n    parameter blank, then you may receive the status of all of them.\n    Please specify the actual machine name for each VM if there are more than one.\n\n    "
    if name:
        vm_ = get_vm_info(name)
        machine = vm_['machine']
        cwd = vm_['cwd'] or cwd
    else:
        if not cwd:
            raise SaltInvocationError(f'Path to Vagranfile must be defined, but cwd={cwd}')
        machine = ''
    info = []
    cmd = f'vagrant status {machine}'
    reply = __salt__['cmd.shell'](cmd, cwd)
    log.info('--->\n%s', reply)
    for line in reply.split('\n'):
        tokens = line.strip().split()
        if len(tokens) > 1 and tokens[-1].endswith(')'):
            try:
                datum = {'machine': tokens[0], 'state': ' '.join(tokens[1:-1]), 'provider': tokens[-1].lstrip('(').rstrip(')'), 'name': get_machine_id(tokens[0], cwd)}
                info.append(datum)
            except IndexError:
                pass
    return info

def init(name, cwd=None, machine='', runas=None, start=False, vagrant_provider='', vm=None):
    if False:
        i = 10
        return i + 15
    '\n    Initialize a new Vagrant VM.\n\n    This inputs all the information needed to start a Vagrant VM.  These settings are stored in\n    a Salt sdb database on the Vagrant host minion and used to start, control, and query the\n    guest VMs. The salt_id assigned here is the key field for that database and must be unique.\n\n    :param name: The salt_id name you will use to control this VM\n    :param cwd: The path to the directory where the Vagrantfile is located\n    :param machine: The machine name in the Vagrantfile. If blank, the primary machine will be used.\n    :param runas: The username on the host who owns the Vagrant work files.\n    :param start: (default: False) Start the virtual machine now.\n    :param vagrant_provider: The name of a Vagrant VM provider (if not the default).\n    :param vm: Optionally, all the above information may be supplied in this dictionary.\n    :return: A string indicating success, or False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.init <salt_id> /path/to/Vagrantfile\n        salt my_laptop vagrant.init x1 /projects/bevy_master machine=quail1\n    '
    vm_ = {} if vm is None else vm.copy()
    vm_['name'] = name
    vm_['cwd'] = cwd or vm_.get('cwd')
    if not vm_['cwd']:
        raise SaltInvocationError('Path to Vagrantfile must be defined by "cwd" argument')
    vm_['machine'] = machine or vm_.get('machine', machine)
    vm_['runas'] = runas or vm_.get('runas', runas)
    vm_['vagrant_provider'] = vagrant_provider or vm_.get('vagrant_provider', '')
    _update_vm_info(name, vm_)
    if start:
        log.debug('Starting VM %s', name)
        ret = _start(name, vm_)
    else:
        ret = 'Name {} defined using VM {}'.format(name, vm_['machine'] or '(default)')
    return ret

def start(name):
    if False:
        while True:
            i = 10
    '\n    Start (vagrant up) a virtual machine defined by salt_id name.\n    The machine must have been previously defined using "vagrant.init".\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.start <salt_id>\n    '
    vm_ = get_vm_info(name)
    return _start(name, vm_)

def _start(name, vm_):
    if False:
        while True:
            i = 10
    try:
        machine = vm_['machine']
    except KeyError:
        raise SaltInvocationError(f'No Vagrant machine defined for Salt_id {name}')
    vagrant_provider = vm_.get('vagrant_provider', '')
    provider_ = f'--provider={vagrant_provider}' if vagrant_provider else ''
    cmd = f'vagrant up {machine} {provider_}'
    ret = __salt__['cmd.run_all'](cmd, runas=vm_.get('runas'), cwd=vm_.get('cwd'), output_loglevel='info')
    if machine == '':
        for line in ret['stdout'].split('\n'):
            if line.startswith('==>'):
                machine = line.split()[1].rstrip(':')
                vm_['machine'] = machine
                _update_vm_info(name, vm_)
                break
    if ret['retcode'] == 0:
        return f'Started "{name}" using Vagrant machine "{machine}".'
    return False

def shutdown(name):
    if False:
        i = 10
        return i + 15
    '\n    Send a soft shutdown (vagrant halt) signal to the named vm.\n\n    This does the same thing as vagrant.stop. Other-VM control\n    modules use "stop" and "shutdown" to differentiate between\n    hard and soft shutdowns.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.shutdown <salt_id>\n    '
    return stop(name)

def stop(name):
    if False:
        print('Hello World!')
    '\n    Hard shutdown the virtual machine. (vagrant halt)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.stop <salt_id>\n    '
    vm_ = get_vm_info(name)
    machine = vm_['machine']
    cmd = f'vagrant halt {machine}'
    ret = __salt__['cmd.retcode'](cmd, runas=vm_.get('runas'), cwd=vm_.get('cwd'))
    return ret == 0

def pause(name):
    if False:
        return 10
    '\n    Pause (vagrant suspend) the named VM.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.pause <salt_id>\n    '
    vm_ = get_vm_info(name)
    machine = vm_['machine']
    cmd = f'vagrant suspend {machine}'
    ret = __salt__['cmd.retcode'](cmd, runas=vm_.get('runas'), cwd=vm_.get('cwd'))
    return ret == 0

def reboot(name, provision=False):
    if False:
        return 10
    '\n    Reboot a VM. (vagrant reload)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.reboot <salt_id> provision=True\n\n    :param name: The salt_id name you will use to control this VM\n    :param provision: (False) also re-run the Vagrant provisioning scripts.\n    '
    vm_ = get_vm_info(name)
    machine = vm_['machine']
    prov = '--provision' if provision else ''
    cmd = f'vagrant reload {machine} {prov}'
    ret = __salt__['cmd.retcode'](cmd, runas=vm_.get('runas'), cwd=vm_.get('cwd'))
    return ret == 0

def destroy(name):
    if False:
        return 10
    '\n    Destroy and delete a virtual machine. (vagrant destroy -f)\n\n    This also removes the salt_id name defined by vagrant.init.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.destroy <salt_id>\n    '
    vm_ = get_vm_info(name)
    machine = vm_['machine']
    cmd = f'vagrant destroy -f {machine}'
    ret = __salt__['cmd.run_all'](cmd, runas=vm_.get('runas'), cwd=vm_.get('cwd'), output_loglevel='info')
    if ret['retcode'] == 0:
        _erase_vm_info(name)
        return f'Destroyed VM {name}'
    return False

def get_ssh_config(name, network_mask='', get_private_key=False):
    if False:
        while True:
            i = 10
    '\n    Retrieve hints of how you might connect to a Vagrant VM.\n\n    :param name: the salt_id of the machine\n    :param network_mask: a CIDR mask to search for the VM\'s address\n    :param get_private_key: (default: False) return the key used for ssh login\n    :return: a dict of ssh login information for the VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <host> vagrant.get_ssh_config <salt_id>\n        salt my_laptop vagrant.get_ssh_config quail1 network_mask=10.0.0.0/8 get_private_key=True\n\n    The returned dictionary contains:\n\n    - key_filename:  the name of the private key file on the VM host computer\n    - ssh_username:  the username to be used to log in to the VM\n    - ssh_host:  the IP address used to log in to the VM.  (This will usually be `127.0.0.1`)\n    - ssh_port:  the TCP port used to log in to the VM.  (This will often be `2222`)\n    - \\[ip_address:\\]  (if `network_mask` is defined. see below)\n    - \\[private_key:\\]  (if `get_private_key` is True) the private key for ssh_username\n\n    About `network_mask`:\n\n    Vagrant usually uses a redirected TCP port on its host computer to log in to a VM using ssh.\n    This redirected port and its IP address are "ssh_port" and "ssh_host".  The ssh_host is\n    usually the localhost (127.0.0.1).\n    This makes it impossible for a third machine (such as a salt-cloud master) to contact the VM\n    unless the VM has another network interface defined.  You will usually want a bridged network\n    defined by having a `config.vm.network "public_network"` statement in your `Vagrantfile`.\n\n    The IP address of the bridged adapter will typically be assigned by DHCP and unknown to you,\n    but you should be able to determine what IP network the address will be chosen from.\n    If you enter a CIDR network mask, Salt will attempt to find the VM\'s address for you.\n    The host machine will send an "ip link show" or "ifconfig" command to the VM\n    (using ssh to `ssh_host`:`ssh_port`) and return the IP address of the first interface it\n    can find which matches your mask.\n    '
    vm_ = get_vm_info(name)
    ssh_config = _vagrant_ssh_config(vm_)
    try:
        ans = {'key_filename': ssh_config['IdentityFile'], 'ssh_username': ssh_config['User'], 'ssh_host': ssh_config['HostName'], 'ssh_port': ssh_config['Port']}
    except KeyError:
        raise CommandExecutionError('Insufficient SSH information to contact VM {}. Is it running?'.format(vm_.get('machine', '(default)')))
    if network_mask:
        command = 'ssh -i {IdentityFile} -p {Port} -oStrictHostKeyChecking={StrictHostKeyChecking} -oUserKnownHostsFile={UserKnownHostsFile} -oControlPath=none {User}@{HostName} ip link show'.format(**ssh_config)
        log.info('Trying ssh -p %(Port)s %(User)s@%(HostName)s ip link show', ssh_config)
        reply = __salt__['cmd.shell'](command)
        log.info('--->\n%s', reply)
        target_network_range = ipaddress.ip_network(network_mask, strict=False)
        found_address = None
        for line in reply.split('\n'):
            try:
                tokens = line.replace('addr:', '', 1).split()
                if 'inet' in tokens:
                    nxt = tokens.index('inet') + 1
                    found_address = ipaddress.ip_address(tokens[nxt])
                elif 'inet6' in tokens:
                    nxt = tokens.index('inet6') + 1
                    found_address = ipaddress.ip_address(tokens[nxt].split('/')[0])
                if found_address in target_network_range:
                    ans['ip_address'] = str(found_address)
                    break
            except (IndexError, AttributeError, TypeError):
                pass
        log.info('Network IP address in %s detected as: %s', target_network_range, ans.get('ip_address', '(not found using ip addr show)'))
        if found_address is None:
            command = 'ssh -i {IdentityFile} -p {Port} -oStrictHostKeyChecking={StrictHostKeyChecking} -oUserKnownHostsFile={UserKnownHostsFile} -oControlPath=none {User}@{HostName} ifconfig'.format(**ssh_config)
            log.info('Trying ssh -p %(Port)s %(User)s@%(HostName)s ifconfig', ssh_config)
            reply = __salt__['cmd.shell'](command)
            log.info('ifconfig returned:\n%s', reply)
            target_network_range = ipaddress.ip_network(network_mask, strict=False)
            for line in reply.split('\n'):
                try:
                    tokens = line.replace('addr:', '', 1).split()
                    found_address = None
                    if 'inet' in tokens:
                        nxt = tokens.index('inet') + 1
                        found_address = ipaddress.ip_address(tokens[nxt])
                    elif 'inet6' in tokens:
                        nxt = tokens.index('inet6') + 1
                        found_address = ipaddress.ip_address(tokens[nxt].split('/')[0])
                    if found_address in target_network_range:
                        ans['ip_address'] = str(found_address)
                        break
                except (IndexError, AttributeError, TypeError):
                    pass
            log.info('Network IP address in %s detected as: %s', target_network_range, ans.get('ip_address', '(not found using ifconfig)'))
    if get_private_key:
        try:
            with salt.utils.files.fopen(ssh_config['IdentityFile']) as pks:
                ans['private_key'] = salt.utils.stringutils.to_unicode(pks.read())
        except OSError as e:
            raise CommandExecutionError(f'Error processing Vagrant private key file: {e}')
    return ans