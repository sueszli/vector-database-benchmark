"""
Module for apcupsd
"""
import logging
import salt.utils.decorators as decorators
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'apcups'

@decorators.memoize
def _check_apcaccess():
    if False:
        return 10
    '\n    Looks to see if apcaccess is present on the system\n    '
    return salt.utils.path.which('apcaccess')

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Provides apcupsd only if apcaccess is present\n    '
    if _check_apcaccess():
        return __virtualname__
    return (False, '{} module can only be loaded on when apcupsd is installed'.format(__virtualname__))

def status():
    if False:
        while True:
            i = 10
    "\n    Return apcaccess output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apcups.status\n    "
    ret = {}
    apcaccess = _check_apcaccess()
    res = __salt__['cmd.run_all'](apcaccess)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = 'Something with wrong executing apcaccess, is apcupsd running?'
        return ret
    for line in res['stdout'].splitlines():
        line = line.split(':')
        ret[line[0].strip()] = line[1].strip()
    return ret

def status_load():
    if False:
        print('Hello World!')
    "\n    Return load\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apcups.status_load\n    "
    data = status()
    if 'LOADPCT' in data:
        load = data['LOADPCT'].split()
        if load[1].lower() == 'percent':
            return float(load[0])
    return {'Error': 'Load not available.'}

def status_charge():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return battery charge\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apcups.status_charge\n    "
    data = status()
    if 'BCHARGE' in data:
        charge = data['BCHARGE'].split()
        if charge[1].lower() == 'percent':
            return float(charge[0])
    return {'Error': 'Load not available.'}

def status_battery():
    if False:
        while True:
            i = 10
    "\n    Return true if running on battery power\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apcups.status_battery\n    "
    data = status()
    if 'TONBATT' in data:
        return not data['TONBATT'] == '0 Seconds'
    return {'Error': 'Battery status not available.'}