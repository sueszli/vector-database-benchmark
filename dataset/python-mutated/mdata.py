"""
Module for managaging metadata in SmartOS Zones

.. versionadded:: 2016.3.0

:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:platform:      smartos
"""
import logging
import salt.utils.decorators as decorators
import salt.utils.path
import salt.utils.platform
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list', 'get_': 'get', 'put_': 'put', 'delete_': 'delete'}
__virtualname__ = 'mdata'

@decorators.memoize
def _check_mdata_list():
    if False:
        while True:
            i = 10
    '\n    looks to see if mdata-list is present on the system\n    '
    return salt.utils.path.which('mdata-list')

@decorators.memoize
def _check_mdata_get():
    if False:
        print('Hello World!')
    '\n    looks to see if mdata-get is present on the system\n    '
    return salt.utils.path.which('mdata-get')

@decorators.memoize
def _check_mdata_put():
    if False:
        print('Hello World!')
    '\n    looks to see if mdata-put is present on the system\n    '
    return salt.utils.path.which('mdata-put')

@decorators.memoize
def _check_mdata_delete():
    if False:
        while True:
            i = 10
    '\n    looks to see if mdata-delete is present on the system\n    '
    return salt.utils.path.which('mdata-delete')

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Provides mdata only on SmartOS\n    '
    if _check_mdata_list() and (not salt.utils.platform.is_smartos_globalzone()):
        return __virtualname__
    return (False, f'{__virtualname__} module can only be loaded on SmartOS zones')

def list_():
    if False:
        print('Hello World!')
    "\n    List available metadata\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' mdata.list\n    "
    mdata = _check_mdata_list()
    if mdata:
        cmd = f'{mdata}'
        return __salt__['cmd.run'](cmd, ignore_retcode=True).splitlines()
    return {}

def get_(*keyname):
    if False:
        return 10
    "\n    Get metadata\n\n    keyname : string\n        name of key\n\n    .. note::\n\n        If no keynames are specified, we get all (public) properties\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' mdata.get salt:role\n        salt '*' mdata.get user-script salt:role\n    "
    mdata = _check_mdata_get()
    ret = {}
    if not keyname:
        keyname = list_()
    for k in keyname:
        if mdata:
            cmd = f'{mdata} {k}'
            res = __salt__['cmd.run_all'](cmd, ignore_retcode=True)
            ret[k] = res['stdout'] if res['retcode'] == 0 else ''
        else:
            ret[k] = ''
    return ret

def put_(keyname, val):
    if False:
        i = 10
        return i + 15
    "\n    Put metadata\n\n    prop : string\n        name of property\n    val : string\n        value to set\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' mdata.list\n    "
    mdata = _check_mdata_put()
    ret = {}
    if mdata:
        cmd = f'echo {val} | {mdata} {keyname}'
        ret = __salt__['cmd.run_all'](cmd, python_shell=True, ignore_retcode=True)
    return ret['retcode'] == 0

def delete_(*keyname):
    if False:
        while True:
            i = 10
    "\n    Delete metadata\n\n    prop : string\n        name of property\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' mdata.get salt:role\n        salt '*' mdata.get user-script salt:role\n    "
    mdata = _check_mdata_delete()
    valid_keynames = list_()
    ret = {}
    for k in keyname:
        if mdata and k in valid_keynames:
            cmd = f'{mdata} {k}'
            ret[k] = __salt__['cmd.run_all'](cmd, ignore_retcode=True)['retcode'] == 0
        else:
            ret[k] = True
    return ret