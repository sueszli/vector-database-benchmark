"""
Module for running nictagadm command on SmartOS
:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:depends:       nictagadm binary, dladm binary
:platform:      smartos

.. versionadded:: 2016.11.0

"""
import logging
import salt.utils.path
import salt.utils.platform
log = logging.getLogger(__name__)
__func_alias__ = {'list_nictags': 'list'}
__virtualname__ = 'nictagadm'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Provides nictagadm on SmartOS\n    '
    if salt.utils.platform.is_smartos_globalzone() and salt.utils.path.which('dladm') and salt.utils.path.which('nictagadm'):
        return __virtualname__
    return (False, f'{__virtualname__} module can only be loaded on SmartOS compute nodes')

def list_nictags(include_etherstubs=True):
    if False:
        i = 10
        return i + 15
    "\n    List all nictags\n\n    include_etherstubs : boolean\n        toggle include of etherstubs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nictagadm.list\n    "
    ret = {}
    cmd = 'nictagadm list -d "|" -p{}'.format(' -L' if not include_etherstubs else '')
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = res['stderr'] if 'stderr' in res else 'Failed to get list of nictags.'
    else:
        header = ['name', 'macaddress', 'link', 'type']
        for nictag in res['stdout'].splitlines():
            nictag = nictag.split('|')
            nictag_data = {}
            for field in header:
                nictag_data[field] = nictag[header.index(field)]
            ret[nictag_data['name']] = nictag_data
            del ret[nictag_data['name']]['name']
    return ret

def vms(nictag):
    if False:
        while True:
            i = 10
    "\n    List all vms connect to nictag\n\n    nictag : string\n        name of nictag\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nictagadm.vms admin\n    "
    ret = {}
    cmd = f'nictagadm vms {nictag}'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = res['stderr'] if 'stderr' in res else 'Failed to get list of vms.'
    else:
        ret = res['stdout'].splitlines()
    return ret

def exists(*nictag, **kwargs):
    if False:
        return 10
    "\n    Check if nictags exists\n\n    nictag : string\n        one or more nictags to check\n    verbose : boolean\n        return list of nictags\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nictagadm.exists admin\n    "
    ret = {}
    if not nictag:
        return {'Error': 'Please provide at least one nictag to check.'}
    cmd = 'nictagadm exists -l {}'.format(' '.join(nictag))
    res = __salt__['cmd.run_all'](cmd)
    if not kwargs.get('verbose', False):
        ret = res['retcode'] == 0
    else:
        missing = res['stderr'].splitlines()
        for nt in nictag:
            ret[nt] = nt not in missing
    return ret

def add(name, mac, mtu=1500):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add a new nictag\n\n    name : string\n        name of new nictag\n    mac : string\n        mac of parent interface or 'etherstub' to create a ether stub\n    mtu : int\n        MTU (ignored for etherstubs)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nictagadm.add storage0 etherstub\n        salt '*' nictagadm.add trunk0 'DE:AD:OO:OO:BE:EF' 9000\n    "
    ret = {}
    if mtu > 9000 or mtu < 1500:
        return {'Error': 'mtu must be a value between 1500 and 9000.'}
    if mac != 'etherstub':
        cmd = 'dladm show-phys -m -p -o address'
        res = __salt__['cmd.run_all'](cmd)
        if mac.replace('00', '0') not in res['stdout'].splitlines():
            return {'Error': f'{mac} is not present on this system.'}
    if mac == 'etherstub':
        cmd = f'nictagadm add -l {name}'
        res = __salt__['cmd.run_all'](cmd)
    else:
        cmd = f'nictagadm add -p mtu={mtu},mac={mac} {name}'
        res = __salt__['cmd.run_all'](cmd)
    if res['retcode'] == 0:
        return True
    else:
        return {'Error': 'failed to create nictag.' if 'stderr' not in res and res['stderr'] == '' else res['stderr']}

def update(name, mac=None, mtu=None):
    if False:
        while True:
            i = 10
    "\n    Update a nictag\n\n    name : string\n        name of nictag\n    mac : string\n        optional new mac for nictag\n    mtu : int\n        optional new MTU for nictag\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nictagadm.update trunk mtu=9000\n    "
    ret = {}
    if name not in list_nictags():
        return {'Error': f'nictag {name} does not exists.'}
    if not mtu and (not mac):
        return {'Error': 'please provide either mac or/and mtu.'}
    if mtu:
        if mtu > 9000 or mtu < 1500:
            return {'Error': 'mtu must be a value between 1500 and 9000.'}
    if mac:
        if mac == 'etherstub':
            return {'Error': 'cannot update a nic with "etherstub".'}
        else:
            cmd = 'dladm show-phys -m -p -o address'
            res = __salt__['cmd.run_all'](cmd)
            if mac.replace('00', '0') not in res['stdout'].splitlines():
                return {'Error': f'{mac} is not present on this system.'}
    if mac and mtu:
        properties = f'mtu={mtu},mac={mac}'
    elif mac:
        properties = f'mac={mac}' if mac else ''
    elif mtu:
        properties = f'mtu={mtu}' if mtu else ''
    cmd = f'nictagadm update -p {properties} {name}'
    res = __salt__['cmd.run_all'](cmd)
    if res['retcode'] == 0:
        return True
    else:
        return {'Error': 'failed to update nictag.' if 'stderr' not in res and res['stderr'] == '' else res['stderr']}

def delete(name, force=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete nictag\n\n    name : string\n        nictag to delete\n    force : boolean\n        force delete even if vms attached\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nictagadm.exists admin\n    "
    ret = {}
    if name not in list_nictags():
        return True
    cmd = 'nictagadm delete {}{}'.format('-f ' if force else '', name)
    res = __salt__['cmd.run_all'](cmd)
    if res['retcode'] == 0:
        return True
    else:
        return {'Error': 'failed to delete nictag.' if 'stderr' not in res and res['stderr'] == '' else res['stderr']}