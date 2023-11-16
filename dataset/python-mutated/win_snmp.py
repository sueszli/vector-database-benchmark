"""
Module for managing SNMP service settings on Windows servers.
The Windows feature 'SNMP-Service' must be installed.
"""
import logging
import salt.utils.platform
from salt.exceptions import CommandExecutionError, SaltInvocationError
_HKEY = 'HKLM'
_SNMP_KEY = 'SYSTEM\\CurrentControlSet\\Services\\SNMP\\Parameters'
_AGENT_KEY = '{}\\RFC1156Agent'.format(_SNMP_KEY)
_COMMUNITIES_KEY = '{}\\ValidCommunities'.format(_SNMP_KEY)
_SNMP_GPO_KEY = 'SOFTWARE\\Policies\\SNMP\\Parameters'
_COMMUNITIES_GPO_KEY = '{}\\ValidCommunities'.format(_SNMP_GPO_KEY)
_PERMISSION_TYPES = {'None': 1, 'Notify': 2, 'Read Only': 4, 'Read Write': 8, 'Read Create': 16}
_SERVICE_TYPES = {'None': 0, 'Physical': 1, 'Datalink and subnetwork': 2, 'Internet': 4, 'End-to-end': 8, 'Applications': 64}
_LOG = logging.getLogger(__name__)
__virtualname__ = 'win_snmp'

def __virtual__():
    if False:
        return 10
    '\n    Only works on Windows systems.\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Module win_snmp: Requires Windows')
    if not __utils__['reg.key_exists'](_HKEY, _SNMP_KEY):
        return (False, 'Module win_snmp: SNMP not installed')
    return __virtualname__

def get_agent_service_types():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the sysServices types that can be configured.\n\n    Returns:\n        list: A list of service types.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_snmp.get_agent_service_types\n    "
    return list(_SERVICE_TYPES)

def get_permission_types():
    if False:
        print('Hello World!')
    "\n    Get the permission types that can be configured for communities.\n\n    Returns:\n        list: A list of permission types.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_snmp.get_permission_types\n    "
    return list(_PERMISSION_TYPES)

def get_agent_settings():
    if False:
        i = 10
        return i + 15
    "\n    Determine the value of the SNMP sysContact, sysLocation, and sysServices\n    settings.\n\n    Returns:\n        dict: A dictionary of the agent settings.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_snmp.get_agent_settings\n    "
    ret = dict()
    sorted_types = sorted(_SERVICE_TYPES.items(), key=lambda x: (-x[1], x[0]))
    ret['services'] = list()
    ret['contact'] = __utils__['reg.read_value'](_HKEY, _AGENT_KEY, 'sysContact')['vdata']
    ret['location'] = __utils__['reg.read_value'](_HKEY, _AGENT_KEY, 'sysLocation')['vdata']
    current_bitmask = __utils__['reg.read_value'](_HKEY, _AGENT_KEY, 'sysServices')['vdata']
    if current_bitmask == 0:
        ret['services'].append(sorted_types[-1][0])
    else:
        for (service, bitmask) in sorted_types:
            if current_bitmask is not None and current_bitmask > 0:
                remaining_bitmask = current_bitmask - bitmask
                if remaining_bitmask >= 0:
                    current_bitmask = remaining_bitmask
                    ret['services'].append(service)
            else:
                break
    ret['services'] = sorted(ret['services'])
    return ret

def set_agent_settings(contact=None, location=None, services=None):
    if False:
        i = 10
        return i + 15
    '\n    Manage the SNMP sysContact, sysLocation, and sysServices settings.\n\n    Args:\n        contact (str, optional): The SNMP contact.\n\n        location (str, optional): The SNMP location.\n\n        services (list, optional): A list of selected services. The possible\n            service names can be found via ``win_snmp.get_agent_service_types``.\n            To disable all services pass a list of None, ie: [\'None\']\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' win_snmp.set_agent_settings contact=\'Contact Name\' location=\'Place\' services="[\'Physical\']"\n    '
    if services is not None:
        services = sorted(set(services))
        for service in services:
            if service not in _SERVICE_TYPES:
                message = "Invalid service '{}' specified. Valid services: {}".format(service, get_agent_service_types())
                raise SaltInvocationError(message)
    settings = {'contact': contact, 'location': location, 'services': services}
    current_settings = get_agent_settings()
    if settings == current_settings:
        _LOG.debug('Agent settings already contain the provided values.')
        return True
    if contact is not None:
        if contact != current_settings['contact']:
            __utils__['reg.set_value'](_HKEY, _AGENT_KEY, 'sysContact', contact, 'REG_SZ')
    if location is not None:
        if location != current_settings['location']:
            __utils__['reg.set_value'](_HKEY, _AGENT_KEY, 'sysLocation', location, 'REG_SZ')
    if services is not None:
        if set(services) != set(current_settings['services']):
            vdata = sum((_SERVICE_TYPES[service] for service in services))
            _LOG.debug('Setting sysServices vdata to: %s', vdata)
            __utils__['reg.set_value'](_HKEY, _AGENT_KEY, 'sysServices', vdata, 'REG_DWORD')
    new_settings = get_agent_settings()
    failed_settings = dict()
    for setting in settings:
        if settings[setting] is not None and settings[setting] != new_settings[setting]:
            failed_settings[setting] = settings[setting]
    if failed_settings:
        _LOG.error('Unable to configure agent settings: %s', failed_settings)
        return False
    _LOG.debug('Agent settings configured successfully: %s', settings.keys())
    return True

def get_auth_traps_enabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    Determine whether the host is configured to send authentication traps.\n\n    Returns:\n        bool: True if traps are enabled, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_snmp.get_auth_traps_enabled\n    "
    reg_ret = __utils__['reg.read_value'](_HKEY, _SNMP_KEY, 'EnableAuthenticationTraps')
    if reg_ret['vdata'] == '(value not set)':
        return False
    return bool(reg_ret['vdata'] or 0)

def set_auth_traps_enabled(status=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Manage the sending of authentication traps.\n\n    Args:\n        status (bool): True to enable traps. False to disable.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_snmp.set_auth_traps_enabled status='True'\n    "
    vname = 'EnableAuthenticationTraps'
    current_status = get_auth_traps_enabled()
    if bool(status) == current_status:
        _LOG.debug('%s already contains the provided value.', vname)
        return True
    vdata = int(status)
    __utils__['reg.set_value'](_HKEY, _SNMP_KEY, vname, vdata, 'REG_DWORD')
    new_status = get_auth_traps_enabled()
    if status == new_status:
        _LOG.debug('Setting %s configured successfully: %s', vname, vdata)
        return True
    _LOG.error('Unable to configure %s with value: %s', vname, vdata)
    return False

def get_community_names():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the current accepted SNMP community names and their permissions.\n\n    If community names are being managed by Group Policy, those values will be\n    returned instead like this:\n\n    .. code-block:: bash\n\n        TestCommunity:\n            Managed by GPO\n\n    Community names managed normally will denote the permission instead:\n\n    .. code-block:: bash\n\n        TestCommunity:\n            Read Only\n\n    Returns:\n        dict: A dictionary of community names and permissions.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_snmp.get_community_names\n    "
    ret = dict()
    if __utils__['reg.key_exists'](_HKEY, _COMMUNITIES_GPO_KEY):
        _LOG.debug('Loading communities from Group Policy settings')
        current_values = __utils__['reg.list_values'](_HKEY, _COMMUNITIES_GPO_KEY)
        if isinstance(current_values, list):
            for current_value in current_values:
                if not isinstance(current_value, dict):
                    continue
                ret[current_value['vdata']] = 'Managed by GPO'
    if not ret:
        _LOG.debug('Loading communities from SNMP settings')
        current_values = __utils__['reg.list_values'](_HKEY, _COMMUNITIES_KEY)
        if isinstance(current_values, list):
            for current_value in current_values:
                if not isinstance(current_value, dict):
                    continue
                permissions = ''
                for permission_name in _PERMISSION_TYPES:
                    if current_value['vdata'] == _PERMISSION_TYPES[permission_name]:
                        permissions = permission_name
                        break
                ret[current_value['vname']] = permissions
    if not ret:
        _LOG.debug('Unable to find existing communities.')
    return ret

def set_community_names(communities):
    if False:
        return 10
    '\n    Manage the SNMP accepted community names and their permissions.\n\n    .. note::\n        Settings managed by Group Policy will always take precedence over those\n        set using the SNMP interface. Therefore if this function finds Group\n        Policy settings it will raise a CommandExecutionError\n\n    Args:\n        communities (dict): A dictionary of SNMP community names and\n            permissions. The possible permissions can be found via\n            ``win_snmp.get_permission_types``.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    Raises:\n        CommandExecutionError:\n            If SNMP settings are being managed by Group Policy\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' win_snmp.set_community_names communities="{\'TestCommunity\': \'Read Only\'}\'\n    '
    values = dict()
    if __utils__['reg.key_exists'](_HKEY, _COMMUNITIES_GPO_KEY):
        _LOG.debug('Communities on this system are managed by Group Policy')
        raise CommandExecutionError('Communities on this system are managed by Group Policy')
    current_communities = get_community_names()
    if communities == current_communities:
        _LOG.debug('Communities already contain the provided values.')
        return True
    for vname in communities:
        if not communities[vname]:
            communities[vname] = 'None'
        try:
            vdata = _PERMISSION_TYPES[communities[vname]]
        except KeyError:
            raise SaltInvocationError("Invalid permission '{}' specified. Valid permissions: {}".format(communities[vname], _PERMISSION_TYPES.keys()))
        values[vname] = vdata
    for current_vname in current_communities:
        if current_vname in values:
            if current_communities[current_vname] != values[current_vname]:
                __utils__['reg.set_value'](_HKEY, _COMMUNITIES_KEY, current_vname, values[current_vname], 'REG_DWORD')
        else:
            __utils__['reg.delete_value'](_HKEY, _COMMUNITIES_KEY, current_vname)
    for vname in values:
        if vname not in current_communities:
            __utils__['reg.set_value'](_HKEY, _COMMUNITIES_KEY, vname, values[vname], 'REG_DWORD')
    new_communities = get_community_names()
    failed_communities = dict()
    for new_vname in new_communities:
        if new_vname not in communities:
            failed_communities[new_vname] = None
    for vname in communities:
        if communities[vname] != new_communities[vname]:
            failed_communities[vname] = communities[vname]
    if failed_communities:
        _LOG.error('Unable to configure communities: %s', failed_communities)
        return False
    _LOG.debug('Communities configured successfully: %s', communities.keys())
    return True