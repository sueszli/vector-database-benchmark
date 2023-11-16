"""
Module for configuring Windows Firewall using ``netsh``
"""
import re
import salt.utils.platform
import salt.utils.win_lgpo_netsh
from salt.exceptions import CommandExecutionError
__virtualname__ = 'firewall'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only works on Windows systems\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Module win_firewall: module only available on Windows')
    return __virtualname__

def get_config():
    if False:
        while True:
            i = 10
    "\n    Get the status of all the firewall profiles\n\n    Returns:\n        dict: A dictionary of all profiles on the system\n\n    Raises:\n        CommandExecutionError: If the command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewall.get_config\n    "
    profiles = {}
    curr = None
    cmd = ['netsh', 'advfirewall', 'show', 'allprofiles']
    ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=True)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stdout'])
    for line in ret['stdout'].splitlines():
        if not curr:
            tmp = re.search('(.*) Profile Settings:', line)
            if tmp:
                curr = tmp.group(1)
        elif line.startswith('State'):
            profiles[curr] = line.split()[1] == 'ON'
            curr = None
    return profiles

def disable(profile='allprofiles'):
    if False:
        return 10
    "\n    Disable firewall profile\n\n    Args:\n        profile (Optional[str]): The name of the profile to disable. Default is\n            ``allprofiles``. Valid options are:\n\n            - allprofiles\n            - domainprofile\n            - privateprofile\n            - publicprofile\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: If the command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewall.disable\n    "
    cmd = ['netsh', 'advfirewall', 'set', profile, 'state', 'off']
    ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=True)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stdout'])
    return True

def enable(profile='allprofiles'):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2015.5.0\n\n    Enable firewall profile\n\n    Args:\n        profile (Optional[str]): The name of the profile to enable. Default is\n            ``allprofiles``. Valid options are:\n\n            - allprofiles\n            - domainprofile\n            - privateprofile\n            - publicprofile\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: If the command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewall.enable\n    "
    cmd = ['netsh', 'advfirewall', 'set', profile, 'state', 'on']
    ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=True)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stdout'])
    return True

def get_rule(name='all'):
    if False:
        return 10
    "\n    .. versionadded:: 2015.5.0\n\n    Display all matching rules as specified by name\n\n    Args:\n        name (Optional[str]): The full name of the rule. ``all`` will return all\n            rules. Default is ``all``\n\n    Returns:\n        dict: A dictionary of all rules or rules that match the name exactly\n\n    Raises:\n        CommandExecutionError: If the command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewall.get_rule 'MyAppPort'\n    "
    cmd = ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name={}'.format(name)]
    ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=True)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stdout'])
    return {name: ret['stdout']}

def add_rule(name, localport, protocol='tcp', action='allow', dir='in', remoteip='any'):
    if False:
        return 10
    '\n    .. versionadded:: 2015.5.0\n\n    Add a new inbound or outbound rule to the firewall policy\n\n    Args:\n\n        name (str): The name of the rule. Must be unique and cannot be "all".\n            Required.\n\n        localport (int): The port the rule applies to. Must be a number between\n            0 and 65535. Can be a range. Can specify multiple ports separated by\n            commas. Required.\n\n        protocol (Optional[str]): The protocol. Can be any of the following:\n\n            - A number between 0 and 255\n            - icmpv4\n            - icmpv6\n            - tcp\n            - udp\n            - any\n\n        action (Optional[str]): The action the rule performs. Can be any of the\n            following:\n\n            - allow\n            - block\n            - bypass\n\n        dir (Optional[str]): The direction. Can be ``in`` or ``out``.\n\n        remoteip (Optional [str]): The remote IP. Can be any of the following:\n\n            - any\n            - localsubnet\n            - dns\n            - dhcp\n            - wins\n            - defaultgateway\n            - Any valid IPv4 address (192.168.0.12)\n            - Any valid IPv6 address (2002:9b3b:1a31:4:208:74ff:fe39:6c43)\n            - Any valid subnet (192.168.1.0/24)\n            - Any valid range of IP addresses (192.168.0.1-192.168.0.12)\n            - A list of valid IP addresses\n\n            Can be combinations of the above separated by commas.\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: If the command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' firewall.add_rule \'test\' \'8080\' \'tcp\'\n        salt \'*\' firewall.add_rule \'test\' \'1\' \'icmpv4\'\n        salt \'*\' firewall.add_rule \'test_remote_ip\' \'8000\' \'tcp\' \'allow\' \'in\' \'192.168.0.1\'\n    '
    cmd = ['netsh', 'advfirewall', 'firewall', 'add', 'rule', 'name={}'.format(name), 'protocol={}'.format(protocol), 'dir={}'.format(dir), 'action={}'.format(action), 'remoteip={}'.format(remoteip)]
    if protocol is None or ('icmpv4' not in protocol and 'icmpv6' not in protocol):
        cmd.append('localport={}'.format(localport))
    ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=True)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stdout'])
    return True

def delete_rule(name=None, localport=None, protocol=None, dir=None, remoteip=None):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2015.8.0\n\n    Delete an existing firewall rule identified by name and optionally by ports,\n    protocols, direction, and remote IP.\n\n    Args:\n\n        name (str): The name of the rule to delete. If the name ``all`` is used\n            you must specify additional parameters.\n\n        localport (Optional[str]): The port of the rule. If protocol is not\n            specified, protocol will be set to ``tcp``\n\n        protocol (Optional[str]): The protocol of the rule. Default is ``tcp``\n            when ``localport`` is specified\n\n        dir (Optional[str]): The direction of the rule.\n\n        remoteip (Optional[str]): The remote IP of the rule.\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: If the command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Delete incoming tcp port 8080 in the rule named 'test'\n        salt '*' firewall.delete_rule 'test' '8080' 'tcp' 'in'\n\n        # Delete the incoming tcp port 8000 from 192.168.0.1 in the rule named\n        # 'test_remote_ip'\n        salt '*' firewall.delete_rule 'test_remote_ip' '8000' 'tcp' 'in' '192.168.0.1'\n\n        # Delete all rules for local port 80:\n        salt '*' firewall.delete_rule all 80 tcp\n\n        # Delete a rule called 'allow80':\n        salt '*' firewall.delete_rule allow80\n    "
    cmd = ['netsh', 'advfirewall', 'firewall', 'delete', 'rule']
    if name:
        cmd.append('name={}'.format(name))
    if protocol:
        cmd.append('protocol={}'.format(protocol))
    if dir:
        cmd.append('dir={}'.format(dir))
    if remoteip:
        cmd.append('remoteip={}'.format(remoteip))
    if protocol is None or ('icmpv4' not in protocol and 'icmpv6' not in protocol):
        if localport:
            if not protocol:
                cmd.append('protocol=tcp')
            cmd.append('localport={}'.format(localport))
    ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=True)
    if ret['retcode'] != 0:
        raise CommandExecutionError(ret['stdout'])
    return True

def rule_exists(name):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.11.6\n\n    Checks if a firewall rule exists in the firewall policy\n\n    Args:\n        name (str): The name of the rule\n\n    Returns:\n        bool: True if exists, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Is there a rule named RemoteDesktop\n        salt '*' firewall.rule_exists RemoteDesktop\n    "
    try:
        get_rule(name)
        return True
    except CommandExecutionError:
        return False

def get_settings(profile, section, store='local'):
    if False:
        return 10
    '\n    Get the firewall property from the specified profile in the specified store\n    as returned by ``netsh advfirewall``.\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        profile (str):\n            The firewall profile to query. Valid options are:\n\n            - domain\n            - public\n            - private\n\n        section (str):\n            The property to query within the selected profile. Valid options\n            are:\n\n            - firewallpolicy : inbound/outbound behavior\n            - logging : firewall logging settings\n            - settings : firewall properties\n            - state : firewalls state (on | off)\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        dict: A dictionary containing the properties for the specified profile\n\n    Raises:\n        CommandExecutionError: If an error occurs\n        ValueError: If the parameters are incorrect\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Get the inbound/outbound firewall settings for connections on the\n        # local domain profile\n        salt * win_firewall.get_settings domain firewallpolicy\n\n        # Get the inbound/outbound firewall settings for connections on the\n        # domain profile as defined by local group policy\n        salt * win_firewall.get_settings domain firewallpolicy lgpo\n    '
    return salt.utils.win_lgpo_netsh.get_settings(profile=profile, section=section, store=store)

def get_all_settings(domain, store='local'):
    if False:
        print('Hello World!')
    '\n    Gets all the properties for the specified profile in the specified store\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        profile (str):\n            The firewall profile to query. Valid options are:\n\n            - domain\n            - public\n            - private\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        dict: A dictionary containing the specified settings\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Get all firewall settings for connections on the domain profile\n        salt * win_firewall.get_all_settings domain\n\n        # Get all firewall settings for connections on the domain profile as\n        # defined by local group policy\n        salt * win_firewall.get_all_settings domain lgpo\n    '
    return salt.utils.win_lgpo_netsh.get_all_settings(profile=domain, store=store)

def get_all_profiles(store='local'):
    if False:
        return 10
    '\n    Gets all properties for all profiles in the specified store\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        dict: A dictionary containing the specified settings for each profile\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Get all firewall settings for all profiles\n        salt * firewall.get_all_settings\n\n        # Get all firewall settings for all profiles as defined by local group\n        # policy\n\n        salt * firewall.get_all_settings lgpo\n    '
    return salt.utils.win_lgpo_netsh.get_all_profiles(store=store)

def set_firewall_settings(profile, inbound=None, outbound=None, store='local'):
    if False:
        return 10
    "\n    Set the firewall inbound/outbound settings for the specified profile and\n    store\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        profile (str):\n            The firewall profile to query. Valid options are:\n\n            - domain\n            - public\n            - private\n\n        inbound (str):\n            The inbound setting. If ``None`` is passed, the setting will remain\n            unchanged. Valid values are:\n\n            - blockinbound\n            - blockinboundalways\n            - allowinbound\n            - notconfigured\n\n            Default is ``None``\n\n        outbound (str):\n            The outbound setting. If ``None`` is passed, the setting will remain\n            unchanged. Valid values are:\n\n            - allowoutbound\n            - blockoutbound\n            - notconfigured\n\n            Default is ``None``\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        bool: ``True`` if successful\n\n    Raises:\n        CommandExecutionError: If an error occurs\n        ValueError: If the parameters are incorrect\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Set the inbound setting for the domain profile to block inbound\n        # connections\n        salt * firewall.set_firewall_settings domain='domain' inbound='blockinbound'\n\n        # Set the outbound setting for the domain profile to allow outbound\n        # connections\n        salt * firewall.set_firewall_settings domain='domain' outbound='allowoutbound'\n\n        # Set inbound/outbound settings for the domain profile in the group\n        # policy to block inbound and allow outbound\n        salt * firewall.set_firewall_settings domain='domain' inbound='blockinbound' outbound='allowoutbound' store='lgpo'\n    "
    return salt.utils.win_lgpo_netsh.set_firewall_settings(profile=profile, inbound=inbound, outbound=outbound, store=store)

def set_logging_settings(profile, setting, value, store='local'):
    if False:
        print('Hello World!')
    "\n    Configure logging settings for the Windows firewall.\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        profile (str):\n            The firewall profile to configure. Valid options are:\n\n            - domain\n            - public\n            - private\n\n        setting (str):\n            The logging setting to configure. Valid options are:\n\n            - allowedconnections\n            - droppedconnections\n            - filename\n            - maxfilesize\n\n        value (str):\n            The value to apply to the setting. Valid values are dependent upon\n            the setting being configured. Valid options are:\n\n            allowedconnections:\n\n                - enable\n                - disable\n                - notconfigured\n\n            droppedconnections:\n\n                - enable\n                - disable\n                - notconfigured\n\n            filename:\n\n                - Full path and name of the firewall log file\n                - notconfigured\n\n            maxfilesize:\n\n                - 1 - 32767\n                - notconfigured\n\n            .. note::\n                ``notconfigured`` can only be used when using the lgpo store\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        bool: ``True`` if successful\n\n    Raises:\n        CommandExecutionError: If an error occurs\n        ValueError: If the parameters are incorrect\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Log allowed connections and set that in local group policy\n        salt * firewall.set_logging_settings domain allowedconnections enable lgpo\n\n        # Don't log dropped connections\n        salt * firewall.set_logging_settings profile=private setting=droppedconnections value=disable\n\n        # Set the location of the log file\n        salt * firewall.set_logging_settings domain filename C:\\windows\\logs\\firewall.log\n\n        # You can also use environment variables\n        salt * firewall.set_logging_settings domain filename %systemroot%\\system32\\LogFiles\\Firewall\\pfirewall.log\n\n        # Set the max file size of the log to 2048 Kb\n        salt * firewall.set_logging_settings domain maxfilesize 2048\n    "
    return salt.utils.win_lgpo_netsh.set_logging_settings(profile=profile, setting=setting, value=value, store=store)

def set_settings(profile, setting, value, store='local'):
    if False:
        while True:
            i = 10
    '\n    Configure firewall settings.\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        profile (str):\n            The firewall profile to configure. Valid options are:\n\n            - domain\n            - public\n            - private\n\n        setting (str):\n            The firewall setting to configure. Valid options are:\n\n            - localfirewallrules\n            - localconsecrules\n            - inboundusernotification\n            - remotemanagement\n            - unicastresponsetomulticast\n\n        value (str):\n            The value to apply to the setting. Valid options are\n\n            - enable\n            - disable\n            - notconfigured\n\n            .. note::\n                ``notconfigured`` can only be used when using the lgpo store\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        bool: ``True`` if successful\n\n    Raises:\n        CommandExecutionError: If an error occurs\n        ValueError: If the parameters are incorrect\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Merge local rules with those distributed through group policy\n        salt * firewall.set_settings domain localfirewallrules enable\n\n        # Allow remote management of Windows Firewall\n        salt * firewall.set_settings domain remotemanagement enable\n    '
    return salt.utils.win_lgpo_netsh.set_settings(profile=profile, setting=setting, value=value, store=store)

def set_state(profile, state, store='local'):
    if False:
        while True:
            i = 10
    '\n    Configure the firewall state.\n\n    .. versionadded:: 2018.3.4\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        profile (str):\n            The firewall profile to configure. Valid options are:\n\n            - domain\n            - public\n            - private\n\n        state (str):\n            The firewall state. Valid options are:\n\n            - on\n            - off\n            - notconfigured\n\n            .. note::\n                ``notconfigured`` can only be used when using the lgpo store\n\n        store (str):\n            The store to use. This is either the local firewall policy or the\n            policy defined by local group policy. Valid options are:\n\n            - lgpo\n            - local\n\n            Default is ``local``\n\n    Returns:\n        bool: ``True`` if successful\n\n    Raises:\n        CommandExecutionError: If an error occurs\n        ValueError: If the parameters are incorrect\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Turn the firewall off when the domain profile is active\n        salt * firewall.set_state domain off\n\n        # Turn the firewall on when the public profile is active and set that in\n        # the local group policy\n        salt * firewall.set_state public on lgpo\n    '
    return salt.utils.win_lgpo_netsh.set_state(profile=profile, state=state, store=store)