"""

Management of Pure Storage FlashArray

Installation Prerequisites
--------------------------
- You will need the ``purestorage`` python package in your python installation
  path that is running salt.

  .. code-block:: bash

      pip install purestorage

- Configure Pure Storage FlashArray authentication. Use one of the following
  three methods.

  1) From the minion config

  .. code-block:: yaml

        pure_tags:
          fa:
            san_ip: management vip or hostname for the FlashArray
            api_token: A valid api token for the FlashArray being managed

  2) From environment (PUREFA_IP and PUREFA_API)
  3) From the pillar (PUREFA_IP and PUREFA_API)

:maintainer: Simon Dodsley (simon@purestorage.com)
:maturity: new
:requires: purestorage
:platform: all

.. versionadded:: 2018.3.0

"""
import os
import platform
from datetime import datetime
from salt.exceptions import CommandExecutionError
try:
    import purestorage
    HAS_PURESTORAGE = True
except ImportError:
    HAS_PURESTORAGE = False
__docformat__ = 'restructuredtext en'
VERSION = '1.0.0'
USER_AGENT_BASE = 'Salt'
__virtualname__ = 'purefa'
DEFAULT_PASSWORD_SYMBOLS = ('23456789', 'ABCDEFGHJKLMNPQRSTUVWXYZ', 'abcdefghijkmnopqrstuvwxyz')

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Determine whether or not to load this module\n    '
    if HAS_PURESTORAGE:
        return __virtualname__
    return (False, 'purefa execution module not loaded: purestorage python library not available.')

def _get_system():
    if False:
        return 10
    '\n    Get Pure Storage FlashArray configuration\n\n    1) From the minion config\n        pure_tags:\n          fa:\n            san_ip: management vip or hostname for the FlashArray\n            api_token: A valid api token for the FlashArray being managed\n    2) From environment (PUREFA_IP and PUREFA_API)\n    3) From the pillar (PUREFA_IP and PUREFA_API)\n\n    '
    agent = {'base': USER_AGENT_BASE, 'class': __name__, 'version': VERSION, 'platform': platform.platform()}
    user_agent = '{base} {class}/{version} ({platform})'.format(**agent)
    try:
        array = __opts__['pure_tags']['fa'].get('san_ip')
        api = __opts__['pure_tags']['fa'].get('api_token')
        if array and api:
            system = purestorage.FlashArray(array, api_token=api, user_agent=user_agent)
    except (KeyError, NameError, TypeError):
        try:
            san_ip = os.environ.get('PUREFA_IP')
            api_token = os.environ.get('PUREFA_API')
            system = purestorage.FlashArray(san_ip, api_token=api_token, user_agent=user_agent)
        except (ValueError, KeyError, NameError):
            try:
                system = purestorage.FlashArray(__pillar__['PUREFA_IP'], api_token=__pillar__['PUREFA_API'], user_agent=user_agent)
            except (KeyError, NameError):
                raise CommandExecutionError('No Pure Storage FlashArray credentials found.')
    try:
        system.get()
    except Exception:
        raise CommandExecutionError('Pure Storage FlashArray authentication failed.')
    return system

def _get_volume(name, array):
    if False:
        for i in range(10):
            print('nop')
    'Private function to check volume'
    try:
        return array.get_volume(name)
    except purestorage.PureError:
        return None

def _get_snapshot(name, suffix, array):
    if False:
        while True:
            i = 10
    'Private function to check snapshot'
    snapshot = name + '.' + suffix
    try:
        for snap in array.get_volume(name, snap=True):
            if snap['name'] == snapshot:
                return snapshot
    except purestorage.PureError:
        return None

def _get_deleted_volume(name, array):
    if False:
        print('Hello World!')
    'Private function to check deleted volume'
    try:
        return array.get_volume(name, pending='true')
    except purestorage.PureError:
        return None

def _get_pgroup(name, array):
    if False:
        for i in range(10):
            print('nop')
    'Private function to check protection group'
    pgroup = None
    for temp in array.list_pgroups():
        if temp['name'] == name:
            pgroup = temp
            break
    return pgroup

def _get_deleted_pgroup(name, array):
    if False:
        return 10
    'Private function to check deleted protection group'
    try:
        return array.get_pgroup(name, pending='true')
    except purestorage.PureError:
        return None

def _get_hgroup(name, array):
    if False:
        while True:
            i = 10
    'Private function to check hostgroup'
    hostgroup = None
    for temp in array.list_hgroups():
        if temp['name'] == name:
            hostgroup = temp
            break
    return hostgroup

def _get_host(name, array):
    if False:
        while True:
            i = 10
    'Private function to check host'
    host = None
    for temp in array.list_hosts():
        if temp['name'] == name:
            host = temp
            break
    return host

def snap_create(name, suffix=None):
    if False:
        while True:
            i = 10
    "\n\n    Create a volume snapshot on a Pure Storage FlashArray.\n\n    Will return False is volume selected to snap does not exist.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume to snapshot\n    suffix : string\n        if specificed forces snapshot name suffix. If not specified defaults to timestamp.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.snap_create foo\n        salt '*' purefa.snap_create foo suffix=bar\n\n    "
    array = _get_system()
    if suffix is None:
        suffix = 'snap-' + str((datetime.utcnow() - datetime(1970, 1, 1, 0, 0, 0, 0)).total_seconds())
        suffix = suffix.replace('.', '')
    if _get_volume(name, array) is not None:
        try:
            array.create_snapshot(name, suffix=suffix)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def snap_delete(name, suffix=None, eradicate=False):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Delete a volume snapshot on a Pure Storage FlashArray.\n\n    Will return False if selected snapshot does not exist.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    suffix : string\n        name of snapshot\n    eradicate : boolean\n        Eradicate snapshot after deletion if True. Default is False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.snap_delete foo suffix=snap eradicate=True\n\n    "
    array = _get_system()
    if _get_snapshot(name, suffix, array) is not None:
        try:
            snapname = name + '.' + suffix
            array.destroy_volume(snapname)
        except purestorage.PureError:
            return False
        if eradicate is True:
            try:
                array.eradicate_volume(snapname)
                return True
            except purestorage.PureError:
                return False
        else:
            return True
    else:
        return False

def snap_eradicate(name, suffix=None):
    if False:
        while True:
            i = 10
    "\n\n    Eradicate a deleted volume snapshot on a Pure Storage FlashArray.\n\n    Will return False if snapshot is not in a deleted state.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    suffix : string\n        name of snapshot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.snap_eradicate foo suffix=snap\n\n    "
    array = _get_system()
    if _get_snapshot(name, suffix, array) is not None:
        snapname = name + '.' + suffix
        try:
            array.eradicate_volume(snapname)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def volume_create(name, size=None):
    if False:
        while True:
            i = 10
    "\n\n    Create a volume on a Pure Storage FlashArray.\n\n    Will return False if volume already exists.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume (truncated to 63 characters)\n    size : string\n        if specificed capacity of volume. If not specified default to 1G.\n        Refer to Pure Storage documentation for formatting rules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_create foo\n        salt '*' purefa.volume_create foo size=10T\n\n    "
    if len(name) > 63:
        name = name[0:63]
    array = _get_system()
    if _get_volume(name, array) is None:
        if size is None:
            size = '1G'
        try:
            array.create_volume(name, size)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def volume_delete(name, eradicate=False):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Delete a volume on a Pure Storage FlashArray.\n\n    Will return False if volume doesn't exist is already in a deleted state.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    eradicate : boolean\n        Eradicate volume after deletion if True. Default is False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_delete foo eradicate=True\n\n    "
    array = _get_system()
    if _get_volume(name, array) is not None:
        try:
            array.destroy_volume(name)
        except purestorage.PureError:
            return False
        if eradicate is True:
            try:
                array.eradicate_volume(name)
                return True
            except purestorage.PureError:
                return False
        else:
            return True
    else:
        return False

def volume_eradicate(name):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Eradicate a deleted volume on a Pure Storage FlashArray.\n\n    Will return False is volume is not in a deleted state.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_eradicate foo\n\n    "
    array = _get_system()
    if _get_deleted_volume(name, array) is not None:
        try:
            array.eradicate_volume(name)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def volume_extend(name, size):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Extend an existing volume on a Pure Storage FlashArray.\n\n    Will return False if new size is less than or equal to existing size.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    size : string\n        New capacity of volume.\n        Refer to Pure Storage documentation for formatting rules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_extend foo 10T\n\n    "
    array = _get_system()
    vol = _get_volume(name, array)
    if vol is not None:
        if __utils__['stringutils.human_to_bytes'](size) > vol['size']:
            try:
                array.extend_volume(name, size)
                return True
            except purestorage.PureError:
                return False
        else:
            return False
    else:
        return False

def snap_volume_create(name, target, overwrite=False):
    if False:
        return 10
    "\n\n    Create R/W volume from snapshot on a Pure Storage FlashArray.\n\n    Will return False if target volume already exists and\n    overwrite is not specified, or selected snapshot doesn't exist.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume snapshot\n    target : string\n        name of clone volume\n    overwrite : boolean\n        overwrite clone if already exists (default: False)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.snap_volume_create foo.bar clone overwrite=True\n\n    "
    array = _get_system()
    (source, suffix) = name.split('.')
    if _get_snapshot(source, suffix, array) is not None:
        if _get_volume(target, array) is None:
            try:
                array.copy_volume(name, target)
                return True
            except purestorage.PureError:
                return False
        elif overwrite:
            try:
                array.copy_volume(name, target, overwrite=overwrite)
                return True
            except purestorage.PureError:
                return False
        else:
            return False
    else:
        return False

def volume_clone(name, target, overwrite=False):
    if False:
        print('Hello World!')
    "\n\n    Clone an existing volume on a Pure Storage FlashArray.\n\n    Will return False if source volume doesn't exist, or\n    target volume already exists and overwrite not specified.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    target : string\n        name of clone volume\n    overwrite : boolean\n        overwrite clone if already exists (default: False)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_clone foo bar overwrite=True\n\n    "
    array = _get_system()
    if _get_volume(name, array) is not None:
        if _get_volume(target, array) is None:
            try:
                array.copy_volume(name, target)
                return True
            except purestorage.PureError:
                return False
        elif overwrite:
            try:
                array.copy_volume(name, target, overwrite=overwrite)
                return True
            except purestorage.PureError:
                return False
        else:
            return False
    else:
        return False

def volume_attach(name, host):
    if False:
        return 10
    "\n\n    Attach a volume to a host on a Pure Storage FlashArray.\n\n    Host and volume must exist or else will return False.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    host : string\n        name of host\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_attach foo bar\n\n    "
    array = _get_system()
    if _get_volume(name, array) is not None and _get_host(host, array) is not None:
        try:
            array.connect_host(host, name)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def volume_detach(name, host):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Detach a volume from a host on a Pure Storage FlashArray.\n\n    Will return False if either host or volume do not exist, or\n    if selected volume isn't already connected to the host.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of volume\n    host : string\n        name of host\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.volume_detach foo bar\n\n    "
    array = _get_system()
    if _get_volume(name, array) is None or _get_host(host, array) is None:
        return False
    elif _get_volume(name, array) is not None and _get_host(host, array) is not None:
        try:
            array.disconnect_host(host, name)
            return True
        except purestorage.PureError:
            return False

def host_create(name, iqn=None, wwn=None, nqn=None):
    if False:
        return 10
    "\n\n    Add a host on a Pure Storage FlashArray.\n\n    Will return False if host already exists, or the iSCSI or\n    Fibre Channel parameters are not in a valid format.\n    See Pure Storage FlashArray documentation.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of host (truncated to 63 characters)\n    iqn : string\n        iSCSI IQN of host\n    nqn : string\n        NVMeF NQN of host\n        .. versionadded:: 3006.0\n    wwn : string\n        Fibre Channel WWN of host\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.host_create foo iqn='<Valid iSCSI IQN>' wwn='<Valid WWN>' nqn='<Valid NQN>'\n\n    "
    array = _get_system()
    if len(name) > 63:
        name = name[0:63]
    if _get_host(name, array) is None:
        try:
            array.create_host(name)
        except purestorage.PureError:
            return False
        if nqn:
            try:
                array.set_host(name, addnqnlist=[nqn])
            except purestorage.PureError:
                array.delete_host(name)
                return False
        if iqn is not None:
            try:
                array.set_host(name, addiqnlist=[iqn])
            except purestorage.PureError:
                array.delete_host(name)
                return False
        if wwn is not None:
            try:
                array.set_host(name, addwwnlist=[wwn])
            except purestorage.PureError:
                array.delete_host(name)
                return False
    else:
        return False
    return True

def host_update(name, iqn=None, wwn=None, nqn=None):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Update a hosts port definitions on a Pure Storage FlashArray.\n\n    Will return False if new port definitions are already in use\n    by another host, or are not in a valid format.\n    See Pure Storage FlashArray documentation.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of host\n    nqn : string\n        Additional NVMeF NQN of host\n        .. versionadded:: 3006.0\n    iqn : string\n        Additional iSCSI IQN of host\n    wwn : string\n        Additional Fibre Channel WWN of host\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.host_update foo iqn='<Valid iSCSI IQN>' wwn='<Valid WWN>' nqn='<Valid NQN>'\n\n    "
    array = _get_system()
    if _get_host(name, array) is not None:
        if nqn:
            try:
                array.set_host(name, addnqnlist=[nqn])
            except purestorage.PureError:
                return False
        if iqn is not None:
            try:
                array.set_host(name, addiqnlist=[iqn])
            except purestorage.PureError:
                return False
        if wwn is not None:
            try:
                array.set_host(name, addwwnlist=[wwn])
            except purestorage.PureError:
                return False
        return True
    else:
        return False

def host_delete(name):
    if False:
        print('Hello World!')
    "\n\n    Delete a host on a Pure Storage FlashArray (detaches all volumes).\n\n    Will return False if the host doesn't exist.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of host\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.host_delete foo\n\n    "
    array = _get_system()
    if _get_host(name, array) is not None:
        for vol in array.list_host_connections(name):
            try:
                array.disconnect_host(name, vol['vol'])
            except purestorage.PureError:
                return False
        try:
            array.delete_host(name)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def hg_create(name, host=None, volume=None):
    if False:
        while True:
            i = 10
    "\n\n    Create a hostgroup on a Pure Storage FlashArray.\n\n    Will return False if hostgroup already exists, or if\n    named host or volume do not exist.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of hostgroup (truncated to 63 characters)\n    host  : string\n         name of host to add to hostgroup\n    volume : string\n         name of volume to add to hostgroup\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.hg_create foo host=bar volume=vol\n\n    "
    array = _get_system()
    if len(name) > 63:
        name = name[0:63]
    if _get_hgroup(name, array) is None:
        try:
            array.create_hgroup(name)
        except purestorage.PureError:
            return False
        if host is not None:
            if _get_host(host, array):
                try:
                    array.set_hgroup(name, addhostlist=[host])
                except purestorage.PureError:
                    return False
            else:
                hg_delete(name)
                return False
        if volume is not None:
            if _get_volume(volume, array):
                try:
                    array.connect_hgroup(name, volume)
                except purestorage.PureError:
                    hg_delete(name)
                    return False
            else:
                hg_delete(name)
                return False
        return True
    else:
        return False

def hg_update(name, host=None, volume=None):
    if False:
        print('Hello World!')
    "\n\n    Adds entries to a hostgroup on a Pure Storage FlashArray.\n\n    Will return False is hostgroup doesn't exist, or host\n    or volume do not exist.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of hostgroup\n    host  : string\n         name of host to add to hostgroup\n    volume : string\n         name of volume to add to hostgroup\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.hg_update foo host=bar volume=vol\n\n    "
    array = _get_system()
    if _get_hgroup(name, array) is not None:
        if host is not None:
            if _get_host(host, array):
                try:
                    array.set_hgroup(name, addhostlist=[host])
                except purestorage.PureError:
                    return False
            else:
                return False
        if volume is not None:
            if _get_volume(volume, array):
                try:
                    array.connect_hgroup(name, volume)
                except purestorage.PureError:
                    return False
            else:
                return False
        return True
    else:
        return False

def hg_delete(name):
    if False:
        i = 10
        return i + 15
    "\n\n    Delete a hostgroup on a Pure Storage FlashArray (removes all volumes and hosts).\n\n    Will return False is hostgroup is already in a deleted state.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of hostgroup\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.hg_delete foo\n\n    "
    array = _get_system()
    if _get_hgroup(name, array) is not None:
        for vol in array.list_hgroup_connections(name):
            try:
                array.disconnect_hgroup(name, vol['vol'])
            except purestorage.PureError:
                return False
        host = array.get_hgroup(name)
        try:
            array.set_hgroup(name, remhostlist=host['hosts'])
            array.delete_hgroup(name)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def hg_remove(name, volume=None, host=None):
    if False:
        print('Hello World!')
    "\n\n    Remove a host and/or volume from a hostgroup on a Pure Storage FlashArray.\n\n    Will return False is hostgroup does not exist, or named host or volume are\n    not in the hostgroup.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of hostgroup\n    volume : string\n       name of volume to remove from hostgroup\n    host : string\n       name of host to remove from hostgroup\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.hg_remove foo volume=test host=bar\n\n    "
    array = _get_system()
    if _get_hgroup(name, array) is not None:
        if volume is not None:
            if _get_volume(volume, array):
                for temp in array.list_hgroup_connections(name):
                    if temp['vol'] == volume:
                        try:
                            array.disconnect_hgroup(name, volume)
                            return True
                        except purestorage.PureError:
                            return False
                return False
            else:
                return False
        if host is not None:
            if _get_host(host, array):
                temp = _get_host(host, array)
                if temp['hgroup'] == name:
                    try:
                        array.set_hgroup(name, remhostlist=[host])
                        return True
                    except purestorage.PureError:
                        return False
                else:
                    return False
            else:
                return False
        if host is None and volume is None:
            return False
    else:
        return False

def pg_create(name, hostgroup=None, host=None, volume=None, enabled=True):
    if False:
        print('Hello World!')
    "\n\n    Create a protection group on a Pure Storage FlashArray.\n\n    Will return False is the following cases:\n       * Protection Grop already exists\n       * Protection Group in a deleted state\n       * More than one type is specified - protection groups are for only\n         hostgroups, hosts or volumes\n       * Named type for protection group does not exist\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of protection group\n    hostgroup  : string\n         name of hostgroup to add to protection group\n    host  : string\n         name of host to add to protection group\n    volume : string\n         name of volume to add to protection group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.pg_create foo [hostgroup=foo | host=bar | volume=vol] enabled=[true | false]\n\n    "
    array = _get_system()
    if hostgroup is None and host is None and (volume is None):
        if _get_pgroup(name, array) is None:
            try:
                array.create_pgroup(name)
            except purestorage.PureError:
                return False
            try:
                array.set_pgroup(name, snap_enabled=enabled)
                return True
            except purestorage.PureError:
                pg_delete(name)
                return False
        else:
            return False
    elif __utils__['value.xor'](hostgroup, host, volume):
        if _get_pgroup(name, array) is None:
            try:
                array.create_pgroup(name)
            except purestorage.PureError:
                return False
            try:
                array.set_pgroup(name, snap_enabled=enabled)
            except purestorage.PureError:
                pg_delete(name)
                return False
            if hostgroup is not None:
                if _get_hgroup(hostgroup, array) is not None:
                    try:
                        array.set_pgroup(name, addhgrouplist=[hostgroup])
                        return True
                    except purestorage.PureError:
                        pg_delete(name)
                        return False
                else:
                    pg_delete(name)
                    return False
            elif host is not None:
                if _get_host(host, array) is not None:
                    try:
                        array.set_pgroup(name, addhostlist=[host])
                        return True
                    except purestorage.PureError:
                        pg_delete(name)
                        return False
                else:
                    pg_delete(name)
                    return False
            elif volume is not None:
                if _get_volume(volume, array) is not None:
                    try:
                        array.set_pgroup(name, addvollist=[volume])
                        return True
                    except purestorage.PureError:
                        pg_delete(name)
                        return False
                else:
                    pg_delete(name)
                    return False
        else:
            return False
    else:
        return False

def pg_update(name, hostgroup=None, host=None, volume=None):
    if False:
        i = 10
        return i + 15
    "\n\n    Update a protection group on a Pure Storage FlashArray.\n\n    Will return False in the following cases:\n      * Protection group does not exist\n      * Incorrect type selected for current protection group type\n      * Specified type does not exist\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of protection group\n    hostgroup  : string\n         name of hostgroup to add to protection group\n    host  : string\n         name of host to add to protection group\n    volume : string\n         name of volume to add to protection group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.pg_update foo [hostgroup=foo | host=bar | volume=vol]\n\n    "
    array = _get_system()
    pgroup = _get_pgroup(name, array)
    if pgroup is not None:
        if hostgroup is not None and pgroup['hgroups'] is not None:
            if _get_hgroup(hostgroup, array) is not None:
                try:
                    array.add_hgroup(hostgroup, name)
                    return True
                except purestorage.PureError:
                    return False
            else:
                return False
        elif host is not None and pgroup['hosts'] is not None:
            if _get_host(host, array) is not None:
                try:
                    array.add_host(host, name)
                    return True
                except purestorage.PureError:
                    return False
            else:
                return False
        elif volume is not None and pgroup['volumes'] is not None:
            if _get_volume(volume, array) is not None:
                try:
                    array.add_volume(volume, name)
                    return True
                except purestorage.PureError:
                    return False
            else:
                return False
        elif pgroup['hgroups'] is None and pgroup['hosts'] is None and (pgroup['volumes'] is None):
            if hostgroup is not None:
                if _get_hgroup(hostgroup, array) is not None:
                    try:
                        array.set_pgroup(name, addhgrouplist=[hostgroup])
                        return True
                    except purestorage.PureError:
                        return False
                else:
                    return False
            elif host is not None:
                if _get_host(host, array) is not None:
                    try:
                        array.set_pgroup(name, addhostlist=[host])
                        return True
                    except purestorage.PureError:
                        return False
                else:
                    return False
            elif volume is not None:
                if _get_volume(volume, array) is not None:
                    try:
                        array.set_pgroup(name, addvollist=[volume])
                        return True
                    except purestorage.PureError:
                        return False
                else:
                    return False
        else:
            return False
    else:
        return False

def pg_delete(name, eradicate=False):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Delete a protecton group on a Pure Storage FlashArray.\n\n    Will return False if protection group is already in a deleted state.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of protection group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.pg_delete foo\n\n    "
    array = _get_system()
    if _get_pgroup(name, array) is not None:
        try:
            array.destroy_pgroup(name)
        except purestorage.PureError:
            return False
        if eradicate is True:
            try:
                array.eradicate_pgroup(name)
                return True
            except purestorage.PureError:
                return False
        else:
            return True
    else:
        return False

def pg_eradicate(name):
    if False:
        i = 10
        return i + 15
    "\n\n    Eradicate a deleted protecton group on a Pure Storage FlashArray.\n\n    Will return False if protection group is not in a deleted state.\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of protection group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.pg_eradicate foo\n\n    "
    array = _get_system()
    if _get_deleted_pgroup(name, array) is not None:
        try:
            array.eradicate_pgroup(name)
            return True
        except purestorage.PureError:
            return False
    else:
        return False

def pg_remove(name, hostgroup=None, host=None, volume=None):
    if False:
        i = 10
        return i + 15
    "\n\n    Remove a hostgroup, host or volume from a protection group on a Pure Storage FlashArray.\n\n    Will return False in the following cases:\n      * Protection group does not exist\n      * Specified type is not currently associated with the protection group\n\n    .. versionadded:: 2018.3.0\n\n    name : string\n        name of hostgroup\n    hostgroup  : string\n         name of hostgroup to remove from protection group\n    host  : string\n         name of host to remove from hostgroup\n    volume : string\n         name of volume to remove from hostgroup\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' purefa.pg_remove foo [hostgroup=bar | host=test | volume=bar]\n\n    "
    array = _get_system()
    pgroup = _get_pgroup(name, array)
    if pgroup is not None:
        if hostgroup is not None and pgroup['hgroups'] is not None:
            if _get_hgroup(hostgroup, array) is not None:
                try:
                    array.remove_hgroup(hostgroup, name)
                    return True
                except purestorage.PureError:
                    return False
            else:
                return False
        elif host is not None and pgroup['hosts'] is not None:
            if _get_host(host, array) is not None:
                try:
                    array.remove_host(host, name)
                    return True
                except purestorage.PureError:
                    return False
            else:
                return False
        elif volume is not None and pgroup['volumes'] is not None:
            if _get_volume(volume, array) is not None:
                try:
                    array.remove_volume(volume, name)
                    return True
                except purestorage.PureError:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False