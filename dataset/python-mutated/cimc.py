"""
Module to provide Cisco UCS compatibility to Salt

:codeauthor: ``Spencer Ervin <spencer_ervin@hotmail.com>``
:maturity:   new
:depends:    none
:platform:   unix


Configuration
=============
This module accepts connection configuration details either as
parameters, or as configuration settings in pillar as a Salt proxy.
Options passed into opts will be ignored if options are passed into pillar.

.. seealso::
    :py:mod:`Cisco UCS Proxy Module <salt.proxy.cimc>`

About
=====
This execution module was designed to handle connections to a Cisco UCS server.
This module adds support to send connections directly to the device through the
rest API.

"""
import logging
import salt.proxy.cimc
import salt.utils.platform
log = logging.getLogger(__name__)
__virtualname__ = 'cimc'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Will load for the cimc proxy minions.\n    '
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'cimc':
            return __virtualname__
    except KeyError:
        pass
    return (False, 'The cimc execution module can only be loaded for cimc proxy minions.')

def activate_backup_image(reset=False):
    if False:
        while True:
            i = 10
    "\n    Activates the firmware backup image.\n\n    CLI Example:\n\n    Args:\n        reset(bool): Reset the CIMC device on activate.\n\n    .. code-block:: bash\n\n        salt '*' cimc.activate_backup_image\n        salt '*' cimc.activate_backup_image reset=True\n\n    "
    dn = 'sys/rack-unit-1/mgmt/fw-boot-def/bootunit-combined'
    r = 'no'
    if reset is True:
        r = 'yes'
    inconfig = "<firmwareBootUnit dn='sys/rack-unit-1/mgmt/fw-boot-def/bootunit-combined'\n    adminState='trigger' image='backup' resetOnActivate='{}' />".format(r)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def create_user(uid=None, username=None, password=None, priv=None):
    if False:
        while True:
            i = 10
    "\n    Create a CIMC user with username and password.\n\n    Args:\n        uid(int): The user ID slot to create the user account in.\n\n        username(str): The name of the user.\n\n        password(str): The clear text password of the user.\n\n        priv(str): The privilege level of the user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.create_user 11 username=admin password=foobar priv=admin\n\n    "
    if not uid:
        raise salt.exceptions.CommandExecutionError('The user ID must be specified.')
    if not username:
        raise salt.exceptions.CommandExecutionError('The username must be specified.')
    if not password:
        raise salt.exceptions.CommandExecutionError('The password must be specified.')
    if not priv:
        raise salt.exceptions.CommandExecutionError('The privilege level must be specified.')
    dn = 'sys/user-ext/user-{}'.format(uid)
    inconfig = '<aaaUser id="{0}" accountStatus="active" name="{1}" priv="{2}"\n    pwd="{3}"  dn="sys/user-ext/user-{0}"/>'.format(uid, username, priv, password)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def get_bios_defaults():
    if False:
        return 10
    "\n    Get the default values of BIOS tokens.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_bios_defaults\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('biosPlatformDefaults', True)
    return ret

def get_bios_settings():
    if False:
        return 10
    "\n    Get the C240 server BIOS token values.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_bios_settings\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('biosSettings', True)
    return ret

def get_boot_order():
    if False:
        while True:
            i = 10
    "\n    Retrieves the configured boot order table.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_boot_order\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('lsbootDef', True)
    return ret

def get_cpu_details():
    if False:
        return 10
    "\n    Get the CPU product ID details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_cpu_details\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('pidCatalogCpu', True)
    return ret

def get_disks():
    if False:
        i = 10
        return i + 15
    "\n    Get the HDD product ID details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_disks\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('pidCatalogHdd', True)
    return ret

def get_ethernet_interfaces():
    if False:
        print('Hello World!')
    "\n    Get the adapter Ethernet interface details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_ethernet_interfaces\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('adaptorHostEthIf', True)
    return ret

def get_fibre_channel_interfaces():
    if False:
        print('Hello World!')
    "\n    Get the adapter fibre channel interface details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_fibre_channel_interfaces\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('adaptorHostFcIf', True)
    return ret

def get_firmware():
    if False:
        for i in range(10):
            print('nop')
    "\n    Retrieves the current running firmware versions of server components.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_firmware\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('firmwareRunning', False)
    return ret

def get_hostname():
    if False:
        return 10
    "\n    Retrieves the hostname from the device.\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_hostname\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('mgmtIf', True)
    try:
        return ret['outConfigs']['mgmtIf'][0]['hostname']
    except Exception as err:
        return 'Unable to retrieve hostname'

def get_ldap():
    if False:
        print('Hello World!')
    "\n    Retrieves LDAP server details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_ldap\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('aaaLdap', True)
    return ret

def get_management_interface():
    if False:
        print('Hello World!')
    "\n    Retrieve the management interface details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_management_interface\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('mgmtIf', False)
    return ret

def get_memory_token():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the memory RAS BIOS token.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_memory_token\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('biosVfSelectMemoryRASConfiguration', False)
    return ret

def get_memory_unit():
    if False:
        return 10
    "\n    Get the IMM/Memory unit product ID details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_memory_unit\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('pidCatalogDimm', True)
    return ret

def get_network_adapters():
    if False:
        while True:
            i = 10
    "\n    Get the list of network adapters and configuration details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_network_adapters\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('networkAdapterEthIf', True)
    return ret

def get_ntp():
    if False:
        i = 10
        return i + 15
    "\n    Retrieves the current running NTP configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_ntp\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('commNtpProvider', False)
    return ret

def get_pci_adapters():
    if False:
        while True:
            i = 10
    "\n    Get the PCI adapter product ID details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_disks\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('pidCatalogPCIAdapter', True)
    return ret

def get_power_configuration():
    if False:
        i = 10
        return i + 15
    "\n    Get the configuration of the power settings from the device. This is only available\n    on some C-Series servers.\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_power_configuration\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('biosVfResumeOnACPowerLoss', True)
    return ret

def get_power_supplies():
    if False:
        while True:
            i = 10
    "\n    Retrieves the power supply unit details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_power_supplies\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('equipmentPsu', False)
    return ret

def get_snmp_config():
    if False:
        print('Hello World!')
    "\n    Get the snmp configuration details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_snmp_config\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('commSnmp', False)
    return ret

def get_syslog():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the Syslog client-server details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_syslog\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('commSyslogClient', False)
    return ret

def get_syslog_settings():
    if False:
        while True:
            i = 10
    "\n    Get the Syslog configuration settings from the system.\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_syslog_settings\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('commSyslog', False)
    return ret

def get_system_info():
    if False:
        print('Hello World!')
    "\n    Get the system information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_system_info\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('computeRackUnit', False)
    return ret

def get_users():
    if False:
        print('Hello World!')
    "\n    Get the CIMC users.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_users\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('aaaUser', False)
    return ret

def get_vic_adapters():
    if False:
        i = 10
        return i + 15
    "\n    Get the VIC adapter general profile details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_vic_adapters\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('adaptorGenProfile', True)
    return ret

def get_vic_uplinks():
    if False:
        return 10
    "\n    Get the VIC adapter uplink port details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.get_vic_uplinks\n\n    "
    ret = __proxy__['cimc.get_config_resolver_class']('adaptorExtEthIf', True)
    return ret

def mount_share(name=None, remote_share=None, remote_file=None, mount_type='nfs', username=None, password=None):
    if False:
        print('Hello World!')
    "\n    Mounts a remote file through a remote share. Currently, this feature is supported in version 1.5 or greater.\n    The remote share can be either NFS, CIFS, or WWW.\n\n    Some of the advantages of CIMC Mounted vMedia include:\n      Communication between mounted media and target stays local (inside datacenter)\n      Media mounts can be scripted/automated\n      No vKVM requirements for media connection\n      Multiple share types supported\n      Connections supported through all CIMC interfaces\n\n      Note: CIMC Mounted vMedia is enabled through BIOS configuration.\n\n    Args:\n        name(str): The name of the volume on the CIMC device.\n\n        remote_share(str): The file share link that will be used to mount the share. This can be NFS, CIFS, or WWW. This\n        must be the directory path and not the full path to the remote file.\n\n        remote_file(str): The name of the remote file to mount. It must reside within remote_share.\n\n        mount_type(str): The type of share to mount. Valid options are nfs, cifs, and www.\n\n        username(str): An optional requirement to pass credentials to the remote share. If not provided, an\n        unauthenticated connection attempt will be made.\n\n        password(str): An optional requirement to pass a password to the remote share. If not provided, an\n        unauthenticated connection attempt will be made.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.mount_share name=WIN7 remote_share=10.xxx.27.xxx:/nfs remote_file=sl1huu.iso\n\n        salt '*' cimc.mount_share name=WIN7 remote_share=10.xxx.27.xxx:/nfs remote_file=sl1huu.iso username=bob password=badpassword\n\n    "
    if not name:
        raise salt.exceptions.CommandExecutionError('The share name must be specified.')
    if not remote_share:
        raise salt.exceptions.CommandExecutionError('The remote share path must be specified.')
    if not remote_file:
        raise salt.exceptions.CommandExecutionError('The remote file name must be specified.')
    if username and password:
        mount_options = " mountOptions='username={},password={}'".format(username, password)
    else:
        mount_options = ''
    dn = 'sys/svc-ext/vmedia-svc/vmmap-{}'.format(name)
    inconfig = "<commVMediaMap dn='sys/svc-ext/vmedia-svc/vmmap-{}' map='{}'{}\n    remoteFile='{}' remoteShare='{}' status='created'\n    volumeName='Win12' />".format(name, mount_type, mount_options, remote_file, remote_share)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def reboot():
    if False:
        i = 10
        return i + 15
    "\n    Power cycling the server.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.reboot\n\n    "
    dn = 'sys/rack-unit-1'
    inconfig = '<computeRackUnit adminPower="cycle-immediate" dn="sys/rack-unit-1"></computeRackUnit>'
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def set_hostname(hostname=None):
    if False:
        i = 10
        return i + 15
    "\n    Sets the hostname on the server.\n\n    .. versionadded:: 2019.2.0\n\n    Args:\n        hostname(str): The new hostname to set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.set_hostname foobar\n\n    "
    if not hostname:
        raise salt.exceptions.CommandExecutionError('Hostname option must be provided.')
    dn = 'sys/rack-unit-1/mgmt/if-1'
    inconfig = '<mgmtIf dn="sys/rack-unit-1/mgmt/if-1" hostname="{}" ></mgmtIf>'.format(hostname)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    try:
        if ret['outConfig']['mgmtIf'][0]['status'] == 'modified':
            return True
        else:
            return False
    except Exception as err:
        return False

def set_logging_levels(remote=None, local=None):
    if False:
        i = 10
        return i + 15
    "\n    Sets the logging levels of the CIMC devices. The logging levels must match\n    the following options: emergency, alert, critical, error, warning, notice,\n    informational, debug.\n\n    .. versionadded:: 2019.2.0\n\n    Args:\n        remote(str): The logging level for SYSLOG logs.\n\n        local(str): The logging level for the local device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.set_logging_levels remote=error local=notice\n\n    "
    logging_options = ['emergency', 'alert', 'critical', 'error', 'warning', 'notice', 'informational', 'debug']
    query = ''
    if remote:
        if remote in logging_options:
            query += ' remoteSeverity="{}"'.format(remote)
        else:
            raise salt.exceptions.CommandExecutionError('Remote Severity option is not valid.')
    if local:
        if local in logging_options:
            query += ' localSeverity="{}"'.format(local)
        else:
            raise salt.exceptions.CommandExecutionError('Local Severity option is not valid.')
    dn = 'sys/svc-ext/syslog'
    inconfig = '<commSyslog dn="sys/svc-ext/syslog"{} ></commSyslog>'.format(query)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def set_ntp_server(server1='', server2='', server3='', server4=''):
    if False:
        return 10
    "\n    Sets the NTP servers configuration. This will also enable the client NTP service.\n\n    Args:\n        server1(str): The first IP address or FQDN of the NTP servers.\n\n        server2(str): The second IP address or FQDN of the NTP servers.\n\n        server3(str): The third IP address or FQDN of the NTP servers.\n\n        server4(str): The fourth IP address or FQDN of the NTP servers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.set_ntp_server 10.10.10.1\n\n        salt '*' cimc.set_ntp_server 10.10.10.1 foo.bar.com\n\n    "
    dn = 'sys/svc-ext/ntp-svc'
    inconfig = '<commNtpProvider dn="sys/svc-ext/ntp-svc" ntpEnable="yes" ntpServer1="{}" ntpServer2="{}"\n    ntpServer3="{}" ntpServer4="{}"/>'.format(server1, server2, server3, server4)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def set_power_configuration(policy=None, delayType=None, delayValue=None):
    if False:
        while True:
            i = 10
    "\n    Sets the power configuration on the device. This is only available for some\n    C-Series servers.\n\n    .. versionadded:: 2019.2.0\n\n    Args:\n        policy(str): The action to be taken when chassis power is restored after\n        an unexpected power loss. This can be one of the following:\n\n            reset: The server is allowed to boot up normally when power is\n            restored. The server can restart immediately or, optionally, after a\n            fixed or random delay.\n\n            stay-off: The server remains off until it is manually restarted.\n\n            last-state: The server restarts and the system attempts to restore\n            any processes that were running before power was lost.\n\n        delayType(str): If the selected policy is reset, the restart can be\n        delayed with this option. This can be one of the following:\n\n            fixed: The server restarts after a fixed delay.\n\n            random: The server restarts after a random delay.\n\n        delayValue(int): If a fixed delay is selected, once chassis power is\n        restored and the Cisco IMC has finished rebooting, the system waits for\n        the specified number of seconds before restarting the server. Enter an\n        integer between 0 and 240.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.set_power_configuration stay-off\n\n        salt '*' cimc.set_power_configuration reset fixed 0\n\n    "
    query = ''
    if policy == 'reset':
        query = ' vpResumeOnACPowerLoss="reset"'
        if delayType:
            if delayType == 'fixed':
                query += ' delayType="fixed"'
                if delayValue:
                    query += ' delay="{}"'.format(delayValue)
            elif delayType == 'random':
                query += ' delayType="random"'
            else:
                raise salt.exceptions.CommandExecutionError('Invalid delay type entered.')
    elif policy == 'stay-off':
        query = ' vpResumeOnACPowerLoss="reset"'
    elif policy == 'last-state':
        query = ' vpResumeOnACPowerLoss="last-state"'
    else:
        raise salt.exceptions.CommandExecutionError('The power state must be specified.')
    dn = 'sys/rack-unit-1/board/Resume-on-AC-power-loss'
    inconfig = '<biosVfResumeOnACPowerLoss\n    dn="sys/rack-unit-1/board/Resume-on-AC-power-loss"{}>\n    </biosVfResumeOnACPowerLoss>'.format(query)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def set_syslog_server(server=None, type='primary'):
    if False:
        i = 10
        return i + 15
    "\n    Set the SYSLOG server on the host.\n\n    Args:\n        server(str): The hostname or IP address of the SYSLOG server.\n\n        type(str): Specifies the type of SYSLOG server. This can either be primary (default) or secondary.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.set_syslog_server foo.bar.com\n\n        salt '*' cimc.set_syslog_server foo.bar.com primary\n\n        salt '*' cimc.set_syslog_server foo.bar.com secondary\n\n    "
    if not server:
        raise salt.exceptions.CommandExecutionError('The SYSLOG server must be specified.')
    if type == 'primary':
        dn = 'sys/svc-ext/syslog/client-primary'
        inconfig = "<commSyslogClient name='primary' adminState='enabled'  hostname='{}'\n        dn='sys/svc-ext/syslog/client-primary'> </commSyslogClient>".format(server)
    elif type == 'secondary':
        dn = 'sys/svc-ext/syslog/client-secondary'
        inconfig = "<commSyslogClient name='secondary' adminState='enabled'  hostname='{}'\n        dn='sys/svc-ext/syslog/client-secondary'> </commSyslogClient>".format(server)
    else:
        raise salt.exceptions.CommandExecutionError('The SYSLOG type must be either primary or secondary.')
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def set_user(uid=None, username=None, password=None, priv=None, status=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets a CIMC user with specified configurations.\n\n    .. versionadded:: 2019.2.0\n\n    Args:\n        uid(int): The user ID slot to create the user account in.\n\n        username(str): The name of the user.\n\n        password(str): The clear text password of the user.\n\n        priv(str): The privilege level of the user.\n\n        status(str): The account status of the user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.set_user 11 username=admin password=foobar priv=admin active\n\n    "
    conf = ''
    if not uid:
        raise salt.exceptions.CommandExecutionError('The user ID must be specified.')
    if status:
        conf += ' accountStatus="{}"'.format(status)
    if username:
        conf += ' name="{}"'.format(username)
    if priv:
        conf += ' priv="{}"'.format(priv)
    if password:
        conf += ' pwd="{}"'.format(password)
    dn = 'sys/user-ext/user-{}'.format(uid)
    inconfig = '<aaaUser id="{0}"{1} dn="sys/user-ext/user-{0}"/>'.format(uid, conf)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def tftp_update_bios(server=None, path=None):
    if False:
        return 10
    "\n    Update the BIOS firmware through TFTP.\n\n    Args:\n        server(str): The IP address or hostname of the TFTP server.\n\n        path(str): The TFTP path and filename for the BIOS image.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.tftp_update_bios foo.bar.com HP-SL2.cap\n\n    "
    if not server:
        raise salt.exceptions.CommandExecutionError('The server name must be specified.')
    if not path:
        raise salt.exceptions.CommandExecutionError('The TFTP path must be specified.')
    dn = 'sys/rack-unit-1/bios/fw-updatable'
    inconfig = "<firmwareUpdatable adminState='trigger' dn='sys/rack-unit-1/bios/fw-updatable'\n    protocol='tftp' remoteServer='{}' remotePath='{}'\n    type='blade-bios' />".format(server, path)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret

def tftp_update_cimc(server=None, path=None):
    if False:
        while True:
            i = 10
    "\n    Update the CIMC firmware through TFTP.\n\n    Args:\n        server(str): The IP address or hostname of the TFTP server.\n\n        path(str): The TFTP path and filename for the CIMC image.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cimc.tftp_update_cimc foo.bar.com HP-SL2.bin\n\n    "
    if not server:
        raise salt.exceptions.CommandExecutionError('The server name must be specified.')
    if not path:
        raise salt.exceptions.CommandExecutionError('The TFTP path must be specified.')
    dn = 'sys/rack-unit-1/mgmt/fw-updatable'
    inconfig = "<firmwareUpdatable adminState='trigger' dn='sys/rack-unit-1/mgmt/fw-updatable'\n    protocol='tftp' remoteServer='{}' remotePath='{}'\n    type='blade-controller' />".format(server, path)
    ret = __proxy__['cimc.set_config_modify'](dn, inconfig, False)
    return ret