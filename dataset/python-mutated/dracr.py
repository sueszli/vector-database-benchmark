"""
Manage Dell DRAC.

.. versionadded:: 2015.8.2
"""
import logging
import os
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__proxyenabled__ = ['fx2']
try:
    run_all = __salt__['cmd.run_all']
except (NameError, KeyError):
    import salt.modules.cmdmod
    __salt__ = {'cmd.run_all': salt.modules.cmdmod.run_all}

def __virtual__():
    if False:
        return 10
    if salt.utils.path.which('racadm'):
        return True
    return (False, 'The drac execution module cannot be loaded: racadm binary not in path.')

def __parse_drac(output):
    if False:
        i = 10
        return i + 15
    '\n    Parse Dell DRAC output\n    '
    drac = {}
    section = ''
    for i in output.splitlines():
        if i.strip().endswith(':') and '=' not in i:
            section = i[0:-1]
            drac[section] = {}
        if i.rstrip() and '=' in i:
            if section in drac:
                drac[section].update(dict([[prop.strip() for prop in i.split('=')]]))
            else:
                section = i.strip()
                if section not in drac and section:
                    drac[section] = {}
    return drac

def __execute_cmd(command, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        return 10
    '\n    Execute rac commands\n    '
    if module:
        if module.startswith('ALL_'):
            modswitch = '-a ' + module[module.index('_') + 1:len(module)].lower()
        else:
            modswitch = '-m {}'.format(module)
    else:
        modswitch = ''
    if not host:
        cmd = __salt__['cmd.run_all']('racadm {} {}'.format(command, modswitch))
    else:
        cmd = __salt__['cmd.run_all']('racadm -r {} -u {} -p {} {} {}'.format(host, admin_username, admin_password, command, modswitch), output_loglevel='quiet')
    if cmd['retcode'] != 0:
        log.warning('racadm returned an exit code of %s', cmd['retcode'])
        return False
    return True

def __execute_ret(command, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        while True:
            i = 10
    '\n    Execute rac commands\n    '
    if module:
        if module == 'ALL':
            modswitch = '-a '
        else:
            modswitch = '-m {}'.format(module)
    else:
        modswitch = ''
    if not host:
        cmd = __salt__['cmd.run_all']('racadm {} {}'.format(command, modswitch))
    else:
        cmd = __salt__['cmd.run_all']('racadm -r {} -u {} -p {} {} {}'.format(host, admin_username, admin_password, command, modswitch), output_loglevel='quiet')
    if cmd['retcode'] != 0:
        log.warning('racadm returned an exit code of %s', cmd['retcode'])
    else:
        fmtlines = []
        for l in cmd['stdout'].splitlines():
            if l.startswith('Security Alert'):
                continue
            if l.startswith('RAC1168:'):
                break
            if l.startswith('RAC1169:'):
                break
            if l.startswith('Continuing execution'):
                continue
            if not l.strip():
                continue
            fmtlines.append(l)
            if '=' in l:
                continue
        cmd['stdout'] = '\n'.join(fmtlines)
    return cmd

def get_dns_dracname(host=None, admin_username=None, admin_password=None):
    if False:
        return 10
    ret = __execute_ret('get iDRAC.NIC.DNSRacName', host=host, admin_username=admin_username, admin_password=admin_password)
    parsed = __parse_drac(ret['stdout'])
    return parsed

def set_dns_dracname(name, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    ret = __execute_ret('set iDRAC.NIC.DNSRacName {}'.format(name), host=host, admin_username=admin_username, admin_password=admin_password)
    return ret

def system_info(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        return 10
    '\n    Return System information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.system_info\n    '
    cmd = __execute_ret('getsysinfo', host=host, admin_username=admin_username, admin_password=admin_password, module=module)
    if cmd['retcode'] != 0:
        log.warning('racadm returned an exit code of %s', cmd['retcode'])
        return cmd
    return __parse_drac(cmd['stdout'])

def set_niccfg(ip=None, netmask=None, gateway=None, dhcp=False, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        print('Hello World!')
    cmdstr = 'setniccfg '
    if dhcp:
        cmdstr += '-d '
    else:
        cmdstr += '-s ' + ip + ' ' + netmask + ' ' + gateway
    return __execute_cmd(cmdstr, host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def set_nicvlan(vlan=None, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        i = 10
        return i + 15
    cmdstr = 'setniccfg -v '
    if vlan:
        cmdstr += vlan
    ret = __execute_cmd(cmdstr, host=host, admin_username=admin_username, admin_password=admin_password, module=module)
    return ret

def network_info(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        i = 10
        return i + 15
    '\n    Return Network Configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.network_info\n    '
    inv = inventory(host=host, admin_username=admin_username, admin_password=admin_password)
    if inv is None:
        cmd = {}
        cmd['retcode'] = -1
        cmd['stdout'] = 'Problem getting switch inventory'
        return cmd
    if module not in inv.get('switch') and module not in inv.get('server'):
        cmd = {}
        cmd['retcode'] = -1
        cmd['stdout'] = 'No module {} found.'.format(module)
        return cmd
    cmd = __execute_ret('getniccfg', host=host, admin_username=admin_username, admin_password=admin_password, module=module)
    if cmd['retcode'] != 0:
        log.warning('racadm returned an exit code of %s', cmd['retcode'])
    cmd['stdout'] = 'Network:\n' + 'Device = ' + module + '\n' + cmd['stdout']
    return __parse_drac(cmd['stdout'])

def nameservers(ns, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure the nameservers on the DRAC\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.nameservers [NAMESERVERS]\n        salt dell dracr.nameservers ns1.example.com ns2.example.com\n            admin_username=root admin_password=calvin module=server-1\n            host=192.168.1.1\n    '
    if len(ns) > 2:
        log.warning('racadm only supports two nameservers')
        return False
    for i in range(1, len(ns) + 1):
        if not __execute_cmd('config -g cfgLanNetworking -o cfgDNSServer{} {}'.format(i, ns[i - 1]), host=host, admin_username=admin_username, admin_password=admin_password, module=module):
            return False
    return True

def syslog(server, enable=True, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        return 10
    '\n    Configure syslog remote logging, by default syslog will automatically be\n    enabled if a server is specified. However, if you want to disable syslog\n    you will need to specify a server followed by False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.syslog [SYSLOG IP] [ENABLE/DISABLE]\n        salt dell dracr.syslog 0.0.0.0 False\n    '
    if enable and __execute_cmd('config -g cfgRemoteHosts -o cfgRhostsSyslogEnable 1', host=host, admin_username=admin_username, admin_password=admin_password, module=None):
        return __execute_cmd('config -g cfgRemoteHosts -o cfgRhostsSyslogServer1 {}'.format(server), host=host, admin_username=admin_username, admin_password=admin_password, module=module)
    return __execute_cmd('config -g cfgRemoteHosts -o cfgRhostsSyslogEnable 0', host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def email_alerts(action, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    '\n    Enable/Disable email alerts\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.email_alerts True\n        salt dell dracr.email_alerts False\n    '
    if action:
        return __execute_cmd('config -g cfgEmailAlert -o cfgEmailAlertEnable -i 1 1', host=host, admin_username=admin_username, admin_password=admin_password)
    else:
        return __execute_cmd('config -g cfgEmailAlert -o cfgEmailAlertEnable -i 1 0')

def list_users(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        i = 10
        return i + 15
    '\n    List all DRAC users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.list_users\n    '
    users = {}
    _username = ''
    for idx in range(1, 17):
        cmd = __execute_ret('getconfig -g cfgUserAdmin -i {}'.format(idx), host=host, admin_username=admin_username, admin_password=admin_password)
        if cmd['retcode'] != 0:
            log.warning('racadm returned an exit code of %s', cmd['retcode'])
        for user in cmd['stdout'].splitlines():
            if not user.startswith('cfg'):
                continue
            (key, val) = user.split('=')
            if key.startswith('cfgUserAdminUserName'):
                _username = val.strip()
                if val:
                    users[_username] = {'index': idx}
                else:
                    break
            elif _username:
                users[_username].update({key: val})
    return users

def delete_user(username, uid=None, host=None, admin_username=None, admin_password=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.delete_user [USERNAME] [UID - optional]\n        salt dell dracr.delete_user diana 4\n    '
    if uid is None:
        user = list_users()
        uid = user[username]['index']
    if uid:
        return __execute_cmd('config -g cfgUserAdmin -o cfgUserAdminUserName -i {} '.format(uid), host=host, admin_username=admin_username, admin_password=admin_password)
    else:
        log.warning("User '%s' does not exist", username)
        return False

def change_password(username, password, uid=None, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        i = 10
        return i + 15
    "\n    Change user's password\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.change_password [USERNAME] [PASSWORD] uid=[OPTIONAL]\n            host=<remote DRAC> admin_username=<DRAC user>\n            admin_password=<DRAC PW>\n        salt dell dracr.change_password diana secret\n\n    Note that if only a username is specified then this module will look up\n    details for all 16 possible DRAC users.  This is time consuming, but might\n    be necessary if one is not sure which user slot contains the one you want.\n    Many late-model Dell chassis have 'root' as UID 1, so if you can depend\n    on that then setting the password is much quicker.\n    Raises an error if the supplied password is greater than 20 chars.\n    "
    if len(password) > 20:
        raise CommandExecutionError('Supplied password should be 20 characters or less')
    if uid is None:
        user = list_users(host=host, admin_username=admin_username, admin_password=admin_password, module=module)
        uid = user[username]['index']
    if uid:
        return __execute_cmd('config -g cfgUserAdmin -o cfgUserAdminPassword -i {} {}'.format(uid, password), host=host, admin_username=admin_username, admin_password=admin_password, module=module)
    else:
        log.warning("racadm: user '%s' does not exist", username)
        return False

def deploy_password(username, password, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        print('Hello World!')
    "\n    Change the QuickDeploy password, used for switches as well\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.deploy_password [USERNAME] [PASSWORD]\n            host=<remote DRAC> admin_username=<DRAC user>\n            admin_password=<DRAC PW>\n        salt dell dracr.change_password diana secret\n\n    Note that if only a username is specified then this module will look up\n    details for all 16 possible DRAC users.  This is time consuming, but might\n    be necessary if one is not sure which user slot contains the one you want.\n    Many late-model Dell chassis have 'root' as UID 1, so if you can depend\n    on that then setting the password is much quicker.\n    "
    return __execute_cmd('deploy -u {} -p {}'.format(username, password), host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def deploy_snmp(snmp, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        print('Hello World!')
    '\n    Change the QuickDeploy SNMP community string, used for switches as well\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.deploy_snmp SNMP_STRING\n            host=<remote DRAC or CMC> admin_username=<DRAC user>\n            admin_password=<DRAC PW>\n        salt dell dracr.deploy_password diana secret\n\n    '
    return __execute_cmd('deploy -v SNMPv2 {} ro'.format(snmp), host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def create_user(username, password, permissions, users=None, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    '\n    Create user accounts\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.create_user [USERNAME] [PASSWORD] [PRIVILEGES]\n        salt dell dracr.create_user diana secret login,test_alerts,clear_logs\n\n    DRAC Privileges\n      * login                   : Login to iDRAC\n      * drac                    : Configure iDRAC\n      * user_management         : Configure Users\n      * clear_logs              : Clear Logs\n      * server_control_commands : Execute Server Control Commands\n      * console_redirection     : Access Console Redirection\n      * virtual_media           : Access Virtual Media\n      * test_alerts             : Test Alerts\n      * debug_commands          : Execute Debug Commands\n    '
    _uids = set()
    if users is None:
        users = list_users()
    if username in users:
        log.warning("racadm: user '%s' already exists", username)
        return False
    for idx in users.keys():
        _uids.add(users[idx]['index'])
    uid = sorted(list(set(range(2, 12)) - _uids), reverse=True).pop()
    if not __execute_cmd('config -g cfgUserAdmin -o cfgUserAdminUserName -i {} {}'.format(uid, username), host=host, admin_username=admin_username, admin_password=admin_password):
        delete_user(username, uid)
        return False
    if not set_permissions(username, permissions, uid):
        log.warning('unable to set user permissions')
        delete_user(username, uid)
        return False
    if not change_password(username, password, uid):
        log.warning('unable to set user password')
        delete_user(username, uid)
        return False
    if not __execute_cmd('config -g cfgUserAdmin -o cfgUserAdminEnable -i {} 1'.format(uid)):
        delete_user(username, uid)
        return False
    return True

def set_permissions(username, permissions, uid=None, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    '\n    Configure users permissions\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.set_permissions [USERNAME] [PRIVILEGES]\n             [USER INDEX - optional]\n        salt dell dracr.set_permissions diana login,test_alerts,clear_logs 4\n\n    DRAC Privileges\n      * login                   : Login to iDRAC\n      * drac                    : Configure iDRAC\n      * user_management         : Configure Users\n      * clear_logs              : Clear Logs\n      * server_control_commands : Execute Server Control Commands\n      * console_redirection     : Access Console Redirection\n      * virtual_media           : Access Virtual Media\n      * test_alerts             : Test Alerts\n      * debug_commands          : Execute Debug Commands\n    '
    privileges = {'login': '0x0000001', 'drac': '0x0000002', 'user_management': '0x0000004', 'clear_logs': '0x0000008', 'server_control_commands': '0x0000010', 'console_redirection': '0x0000020', 'virtual_media': '0x0000040', 'test_alerts': '0x0000080', 'debug_commands': '0x0000100'}
    permission = 0
    if uid is None:
        user = list_users()
        uid = user[username]['index']
    for i in permissions.split(','):
        perm = i.strip()
        if perm in privileges:
            permission += int(privileges[perm], 16)
    return __execute_cmd('config -g cfgUserAdmin -o cfgUserAdminPrivilege -i {} 0x{:08X}'.format(uid, permission), host=host, admin_username=admin_username, admin_password=admin_password)

def set_snmp(community, host=None, admin_username=None, admin_password=None):
    if False:
        return 10
    '\n    Configure CMC or individual iDRAC SNMP community string.\n    Use ``deploy_snmp`` for configuring chassis switch SNMP.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.set_snmp [COMMUNITY]\n        salt dell dracr.set_snmp public\n    '
    return __execute_cmd('config -g cfgOobSnmp -o cfgOobSnmpAgentCommunity {}'.format(community), host=host, admin_username=admin_username, admin_password=admin_password)

def set_network(ip, netmask, gateway, host=None, admin_username=None, admin_password=None):
    if False:
        return 10
    '\n    Configure Network on the CMC or individual iDRAC.\n    Use ``set_niccfg`` for blade and switch addresses.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.set_network [DRAC IP] [NETMASK] [GATEWAY]\n        salt dell dracr.set_network 192.168.0.2 255.255.255.0 192.168.0.1\n            admin_username=root admin_password=calvin host=192.168.1.1\n    '
    return __execute_cmd('setniccfg -s {} {} {}'.format(ip, netmask, gateway, host=host, admin_username=admin_username, admin_password=admin_password))

def server_power(status, host=None, admin_username=None, admin_password=None, module=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    status\n        One of 'powerup', 'powerdown', 'powercycle', 'hardreset',\n        'graceshutdown'\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    module\n        The element to reboot on the chassis such as a blade. If not provided,\n        the chassis will be rebooted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.server_reboot\n        salt dell dracr.server_reboot module=server-1\n\n    "
    return __execute_cmd('serveraction {}'.format(status), host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def server_reboot(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        i = 10
        return i + 15
    "\n    Issues a power-cycle operation on the managed server. This action is\n    similar to pressing the power button on the system's front panel to\n    power down and then power up the system.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    module\n        The element to reboot on the chassis such as a blade. If not provided,\n        the chassis will be rebooted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.server_reboot\n        salt dell dracr.server_reboot module=server-1\n\n    "
    return __execute_cmd('serveraction powercycle', host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def server_poweroff(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Powers down the managed server.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    module\n        The element to power off on the chassis such as a blade.\n        If not provided, the chassis will be powered off.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.server_poweroff\n        salt dell dracr.server_poweroff module=server-1\n    '
    return __execute_cmd('serveraction powerdown', host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def server_poweron(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        i = 10
        return i + 15
    '\n    Powers up the managed server.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    module\n        The element to power on located on the chassis such as a blade. If\n        not provided, the chassis will be powered on.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.server_poweron\n        salt dell dracr.server_poweron module=server-1\n    '
    return __execute_cmd('serveraction powerup', host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def server_hardreset(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        print('Hello World!')
    '\n    Performs a reset (reboot) operation on the managed server.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    module\n        The element to hard reset on the chassis such as a blade. If\n        not provided, the chassis will be reset.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.server_hardreset\n        salt dell dracr.server_hardreset module=server-1\n    '
    return __execute_cmd('serveraction hardreset', host=host, admin_username=admin_username, admin_password=admin_password, module=module)

def server_powerstatus(host=None, admin_username=None, admin_password=None, module=None):
    if False:
        print('Hello World!')
    '\n    return the power status for the passed module\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell drac.server_powerstatus\n    '
    ret = __execute_ret('serveraction powerstatus', host=host, admin_username=admin_username, admin_password=admin_password, module=module)
    result = {'retcode': 0}
    if ret['stdout'] == 'ON':
        result['status'] = True
        result['comment'] = 'Power is on'
    if ret['stdout'] == 'OFF':
        result['status'] = False
        result['comment'] = 'Power is on'
    if ret['stdout'].startswith('ERROR'):
        result['status'] = False
        result['comment'] = ret['stdout']
    return result

def server_pxe(host=None, admin_username=None, admin_password=None):
    if False:
        while True:
            i = 10
    '\n    Configure server to PXE perform a one off PXE boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt dell dracr.server_pxe\n    '
    if __execute_cmd('config -g cfgServerInfo -o cfgServerFirstBootDevice PXE', host=host, admin_username=admin_username, admin_password=admin_password):
        if __execute_cmd('config -g cfgServerInfo -o cfgServerBootOnce 1', host=host, admin_username=admin_username, admin_password=admin_password):
            return server_reboot
        else:
            log.warning('failed to set boot order')
            return False
    log.warning('failed to configure PXE boot')
    return False

def list_slotnames(host=None, admin_username=None, admin_password=None):
    if False:
        print('Hello World!')
    '\n    List the names of all slots in the chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call --local dracr.list_slotnames host=111.222.333.444\n            admin_username=root admin_password=secret\n\n    '
    slotraw = __execute_ret('getslotname', host=host, admin_username=admin_username, admin_password=admin_password)
    if slotraw['retcode'] != 0:
        return slotraw
    slots = {}
    stripheader = True
    for l in slotraw['stdout'].splitlines():
        if l.startswith('<'):
            stripheader = False
            continue
        if stripheader:
            continue
        fields = l.split()
        slots[fields[0]] = {}
        slots[fields[0]]['slot'] = fields[0]
        if len(fields) > 1:
            slots[fields[0]]['slotname'] = fields[1]
        else:
            slots[fields[0]]['slotname'] = ''
        if len(fields) > 2:
            slots[fields[0]]['hostname'] = fields[2]
        else:
            slots[fields[0]]['hostname'] = ''
    return slots

def get_slotname(slot, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    '\n    Get the name of a slot number in the chassis.\n\n    slot\n        The number of the slot for which to obtain the name.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call --local dracr.get_slotname 0 host=111.222.333.444\n           admin_username=root admin_password=secret\n\n    '
    slots = list_slotnames(host=host, admin_username=admin_username, admin_password=admin_password)
    slot = str(slot)
    return slots[slot]['slotname']

def set_slotname(slot, name, host=None, admin_username=None, admin_password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the name of a slot in a chassis.\n\n    slot\n        The slot number to change.\n\n    name\n        The name to set. Can only be 15 characters long.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.set_slotname 2 my-slotname host=111.222.333.444\n            admin_username=root admin_password=secret\n\n    "
    return __execute_cmd('config -g cfgServerInfo -o cfgServerName -i {} {}'.format(slot, name), host=host, admin_username=admin_username, admin_password=admin_password)

def set_chassis_name(name, host=None, admin_username=None, admin_password=None):
    if False:
        while True:
            i = 10
    "\n    Set the name of the chassis.\n\n    name\n        The name to be set on the chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.set_chassis_name my-chassis host=111.222.333.444\n            admin_username=root admin_password=secret\n\n    "
    return __execute_cmd('setsysinfo -c chassisname {}'.format(name), host=host, admin_username=admin_username, admin_password=admin_password)

def get_chassis_name(host=None, admin_username=None, admin_password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the name of a chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.get_chassis_name host=111.222.333.444\n            admin_username=root admin_password=secret\n\n    "
    return bare_rac_cmd('getchassisname', host=host, admin_username=admin_username, admin_password=admin_password)

def inventory(host=None, admin_username=None, admin_password=None):
    if False:
        return 10

    def mapit(x, y):
        if False:
            i = 10
            return i + 15
        return {x: y}
    fields = {}
    fields['server'] = ['name', 'idrac_version', 'blade_type', 'gen', 'updateable']
    fields['switch'] = ['name', 'model_name', 'hw_version', 'fw_version']
    fields['cmc'] = ['name', 'cmc_version', 'updateable']
    fields['chassis'] = ['name', 'fw_version', 'fqdd']
    rawinv = __execute_ret('getversion', host=host, admin_username=admin_username, admin_password=admin_password)
    if rawinv['retcode'] != 0:
        return rawinv
    in_server = False
    in_switch = False
    in_cmc = False
    in_chassis = False
    ret = {}
    ret['server'] = {}
    ret['switch'] = {}
    ret['cmc'] = {}
    ret['chassis'] = {}
    for l in rawinv['stdout'].splitlines():
        if l.startswith('<Server>'):
            in_server = True
            in_switch = False
            in_cmc = False
            in_chassis = False
            continue
        if l.startswith('<Switch>'):
            in_server = False
            in_switch = True
            in_cmc = False
            in_chassis = False
            continue
        if l.startswith('<CMC>'):
            in_server = False
            in_switch = False
            in_cmc = True
            in_chassis = False
            continue
        if l.startswith('<Chassis Infrastructure>'):
            in_server = False
            in_switch = False
            in_cmc = False
            in_chassis = True
            continue
        if not l:
            continue
        line = re.split('  +', l.strip())
        if in_server:
            ret['server'][line[0]] = {k: v for d in map(mapit, fields['server'], line) for (k, v) in d.items()}
        if in_switch:
            ret['switch'][line[0]] = {k: v for d in map(mapit, fields['switch'], line) for (k, v) in d.items()}
        if in_cmc:
            ret['cmc'][line[0]] = {k: v for d in map(mapit, fields['cmc'], line) for (k, v) in d.items()}
        if in_chassis:
            ret['chassis'][line[0]] = {k: v for d in map(mapit, fields['chassis'], line) for (k, v) in d.items()}
    return ret

def set_chassis_location(location, host=None, admin_username=None, admin_password=None):
    if False:
        while True:
            i = 10
    "\n    Set the location of the chassis.\n\n    location\n        The name of the location to be set on the chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.set_chassis_location location-name host=111.222.333.444\n            admin_username=root admin_password=secret\n\n    "
    return __execute_cmd('setsysinfo -c chassislocation {}'.format(location), host=host, admin_username=admin_username, admin_password=admin_password)

def get_chassis_location(host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get the location of the chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.set_chassis_location host=111.222.333.444\n           admin_username=root admin_password=secret\n\n    "
    return system_info(host=host, admin_username=admin_username, admin_password=admin_password)['Chassis Information']['Chassis Location']

def set_chassis_datacenter(location, host=None, admin_username=None, admin_password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the location of the chassis.\n\n    location\n        The name of the datacenter to be set on the chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.set_chassis_datacenter datacenter-name host=111.222.333.444\n            admin_username=root admin_password=secret\n\n    "
    return set_general('cfgLocation', 'cfgLocationDatacenter', location, host=host, admin_username=admin_username, admin_password=admin_password)

def get_chassis_datacenter(host=None, admin_username=None, admin_password=None):
    if False:
        print('Hello World!')
    "\n    Get the datacenter of the chassis.\n\n    host\n        The chassis host.\n\n    admin_username\n        The username used to access the chassis.\n\n    admin_password\n        The password used to access the chassis.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dracr.set_chassis_location host=111.222.333.444\n           admin_username=root admin_password=secret\n\n    "
    return get_general('cfgLocation', 'cfgLocationDatacenter', host=host, admin_username=admin_username, admin_password=admin_password)

def set_general(cfg_sec, cfg_var, val, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    return __execute_cmd('config -g {} -o {} {}'.format(cfg_sec, cfg_var, val), host=host, admin_username=admin_username, admin_password=admin_password)

def get_general(cfg_sec, cfg_var, host=None, admin_username=None, admin_password=None):
    if False:
        i = 10
        return i + 15
    ret = __execute_ret('getconfig -g {} -o {}'.format(cfg_sec, cfg_var), host=host, admin_username=admin_username, admin_password=admin_password)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        return ret

def idrac_general(blade_name, command, idrac_password=None, host=None, admin_username=None, admin_password=None):
    if False:
        return 10
    "\n    Run a generic racadm command against a particular\n    blade in a chassis.  Blades are usually named things like\n    'server-1', 'server-2', etc.  If the iDRAC has a different\n    password than the CMC, then you can pass it with the\n    idrac_password kwarg.\n\n    :param blade_name: Name of the blade to run the command on\n    :param command: Command like to pass to racadm\n    :param idrac_password: Password for the iDRAC if different from the CMC\n    :param host: Chassis hostname\n    :param admin_username: CMC username\n    :param admin_password: CMC password\n    :return: stdout if the retcode is 0, otherwise a standard cmd.run_all dictionary\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt fx2 chassis.cmd idrac_general server-1 'get BIOS.SysProfileSettings'\n\n    "
    module_network = network_info(host, admin_username, admin_password, blade_name)
    if idrac_password is not None:
        password = idrac_password
    else:
        password = admin_password
    idrac_ip = module_network['Network']['IP Address']
    ret = __execute_ret(command, host=idrac_ip, admin_username='root', admin_password=password)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        return ret

def _update_firmware(cmd, host=None, admin_username=None, admin_password=None):
    if False:
        while True:
            i = 10
    if not admin_username:
        admin_username = __pillar__['proxy']['admin_username']
    if not admin_username:
        admin_password = __pillar__['proxy']['admin_password']
    ret = __execute_ret(cmd, host=host, admin_username=admin_username, admin_password=admin_password)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        return ret

def bare_rac_cmd(cmd, host=None, admin_username=None, admin_password=None):
    if False:
        print('Hello World!')
    ret = __execute_ret('{}'.format(cmd), host=host, admin_username=admin_username, admin_password=admin_password)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        return ret

def update_firmware(filename, host=None, admin_username=None, admin_password=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Updates firmware using local firmware file\n\n    .. code-block:: bash\n\n         salt dell dracr.update_firmware firmware.exe\n\n    This executes the following command on your FX2\n    (using username and password stored in the pillar data)\n\n    .. code-block:: bash\n\n         racadm update –f firmware.exe -u user –p pass\n\n    '
    if os.path.exists(filename):
        return _update_firmware('update -f {}'.format(filename), host=None, admin_username=None, admin_password=None)
    else:
        raise CommandExecutionError('Unable to find firmware file {}'.format(filename))

def update_firmware_nfs_or_cifs(filename, share, host=None, admin_username=None, admin_password=None):
    if False:
        while True:
            i = 10
    '\n    Executes the following for CIFS\n    (using username and password stored in the pillar data)\n\n    .. code-block:: bash\n\n         racadm update -f <updatefile> -u user –p pass -l //IP-Address/share\n\n    Or for NFS\n    (using username and password stored in the pillar data)\n\n    .. code-block:: bash\n\n          racadm update -f <updatefile> -u user –p pass -l IP-address:/share\n\n\n    Salt command for CIFS:\n\n    .. code-block:: bash\n\n         salt dell dracr.update_firmware_nfs_or_cifs          firmware.exe //IP-Address/share\n\n\n    Salt command for NFS:\n\n    .. code-block:: bash\n\n         salt dell dracr.update_firmware_nfs_or_cifs          firmware.exe IP-address:/share\n    '
    if os.path.exists(filename):
        return _update_firmware('update -f {} -l {}'.format(filename, share), host=None, admin_username=None, admin_password=None)
    else:
        raise CommandExecutionError('Unable to find firmware file {}'.format(filename))