"""
Module to provide Palo Alto compatibility to Salt

:codeauthor: ``Spencer Ervin <spencer_ervin@hotmail.com>``
:maturity:   new
:depends:    none
:platform:   unix

.. versionadded:: 2018.3.0

Configuration
=============

This module accepts connection configuration details either as
parameters, or as configuration settings in pillar as a Salt proxy.
Options passed into opts will be ignored if options are passed into pillar.

.. seealso::
    :py:mod:`Palo Alto Proxy Module <salt.proxy.panos>`

About
=====

This execution module was designed to handle connections to a Palo Alto based
firewall. This module adds support to send connections directly to the device
through the XML API or through a brokered connection to Panorama.

"""
import logging
import time
import salt.proxy.panos
import salt.utils.platform
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'panos'

def __virtual__():
    if False:
        return 10
    '\n    Will load for the panos proxy minions.\n    '
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'panos':
            return __virtualname__
    except KeyError:
        pass
    return (False, 'The panos execution module can only be loaded for panos proxy minions.')

def _get_job_results(query=None):
    if False:
        while True:
            i = 10
    '\n    Executes a query that requires a job for completion. This function will wait for the job to complete\n    and return the results.\n    '
    if not query:
        raise CommandExecutionError('Query parameters cannot be empty.')
    response = __proxy__['panos.call'](query)
    if 'result' in response and 'job' in response['result']:
        jid = response['result']['job']
        while get_job(jid)['result']['job']['status'] != 'FIN':
            time.sleep(5)
        return get_job(jid)
    else:
        return response

def add_config_lock():
    if False:
        while True:
            i = 10
    "\n    Prevent other users from changing configuration until the lock is released.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.add_config_lock\n\n    "
    query = {'type': 'op', 'cmd': '<request><config-lock><add></add></config-lock></request>'}
    return __proxy__['panos.call'](query)

def check_antivirus():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get anti-virus information from PaloAlto Networks server\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.check_antivirus\n\n    "
    query = {'type': 'op', 'cmd': '<request><anti-virus><upgrade><check></check></upgrade></anti-virus></request>'}
    return __proxy__['panos.call'](query)

def check_software():
    if False:
        i = 10
        return i + 15
    "\n    Get software information from PaloAlto Networks server.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.check_software\n\n    "
    query = {'type': 'op', 'cmd': '<request><system><software><check></check></software></system></request>'}
    return __proxy__['panos.call'](query)

def clear_commit_tasks():
    if False:
        return 10
    "\n    Clear all commit tasks.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.clear_commit_tasks\n\n    "
    query = {'type': 'op', 'cmd': '<request><clear-commit-tasks></clear-commit-tasks></request>'}
    return __proxy__['panos.call'](query)

def commit():
    if False:
        return 10
    "\n    Commits the candidate configuration to the running configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.commit\n\n    "
    query = {'type': 'commit', 'cmd': '<commit></commit>'}
    return _get_job_results(query)

def deactivate_license(key_name=None):
    if False:
        return 10
    "\n    Deactivates an installed license.\n    Required version 7.0.0 or greater.\n\n    key_name(str): The file name of the license key installed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.deactivate_license key_name=License_File_Name.key\n\n    "
    _required_version = '7.0.0'
    if not __proxy__['panos.is_required_version'](_required_version):
        return (False, 'The panos device requires version {} or greater for this command.'.format(_required_version))
    if not key_name:
        return (False, 'You must specify a key_name.')
    else:
        query = {'type': 'op', 'cmd': '<request><license><deactivate><key><features><member>{}</member></features></key></deactivate></license></request>'.format(key_name)}
    return __proxy__['panos.call'](query)

def delete_license(key_name=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove license keys on disk.\n\n    key_name(str): The file name of the license key to be deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.delete_license key_name=License_File_Name.key\n\n    "
    if not key_name:
        return (False, 'You must specify a key_name.')
    else:
        query = {'type': 'op', 'cmd': '<delete><license><key>{}</key></license></delete>'.format(key_name)}
    return __proxy__['panos.call'](query)

def download_antivirus():
    if False:
        for i in range(10):
            print('nop')
    "\n    Download the most recent anti-virus package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.download_antivirus\n\n    "
    query = {'type': 'op', 'cmd': '<request><anti-virus><upgrade><download><latest></latest></download></upgrade></anti-virus></request>'}
    return _get_job_results(query)

def download_software_file(filename=None, synch=False):
    if False:
        print('Hello World!')
    "\n    Download software packages by filename.\n\n    Args:\n        filename(str): The filename of the PANOS file to download.\n\n        synch (bool): If true then the file will synch to the peer unit.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.download_software_file PanOS_5000-8.0.0\n        salt '*' panos.download_software_file PanOS_5000-8.0.0 True\n\n    "
    if not filename:
        raise CommandExecutionError('Filename option must not be none.')
    if not isinstance(synch, bool):
        raise CommandExecutionError('Synch option must be boolean..')
    if synch is True:
        query = {'type': 'op', 'cmd': '<request><system><software><download><file>{}</file></download></software></system></request>'.format(filename)}
    else:
        query = {'type': 'op', 'cmd': '<request><system><software><download><sync-to-peer>yes</sync-to-peer><file>{}</file></download></software></system></request>'.format(filename)}
    return _get_job_results(query)

def download_software_version(version=None, synch=False):
    if False:
        i = 10
        return i + 15
    "\n    Download software packages by version number.\n\n    Args:\n        version(str): The version of the PANOS file to download.\n\n        synch (bool): If true then the file will synch to the peer unit.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.download_software_version 8.0.0\n        salt '*' panos.download_software_version 8.0.0 True\n\n    "
    if not version:
        raise CommandExecutionError('Version option must not be none.')
    if not isinstance(synch, bool):
        raise CommandExecutionError('Synch option must be boolean..')
    if synch is True:
        query = {'type': 'op', 'cmd': '<request><system><software><download><version>{}</version></download></software></system></request>'.format(version)}
    else:
        query = {'type': 'op', 'cmd': '<request><system><software><download><sync-to-peer>yes</sync-to-peer><version>{}</version></download></software></system></request>'.format(version)}
    return _get_job_results(query)

def fetch_license(auth_code=None):
    if False:
        while True:
            i = 10
    "\n    Get new license(s) using from the Palo Alto Network Server.\n\n    auth_code\n        The license authorization code.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.fetch_license\n        salt '*' panos.fetch_license auth_code=foobar\n\n    "
    if not auth_code:
        query = {'type': 'op', 'cmd': '<request><license><fetch></fetch></license></request>'}
    else:
        query = {'type': 'op', 'cmd': '<request><license><fetch><auth-code>{}</auth-code></fetch></license></request>'.format(auth_code)}
    return __proxy__['panos.call'](query)

def get_address(address=None, vsys='1'):
    if False:
        print('Hello World!')
    "\n    Get the candidate configuration for the specified get_address object. This will not return address objects that are\n    marked as pre-defined objects.\n\n    address(str): The name of the address object.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_address myhost\n        salt '*' panos.get_address myhost 3\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/address/entry[@name='{}']".format(vsys, address)}
    return __proxy__['panos.call'](query)

def get_address_group(addressgroup=None, vsys='1'):
    if False:
        return 10
    "\n    Get the candidate configuration for the specified address group. This will not return address groups that are\n    marked as pre-defined objects.\n\n    addressgroup(str): The name of the address group.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_address_group foobar\n        salt '*' panos.get_address_group foobar 3\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/address-group/entry[@name='{}']".format(vsys, addressgroup)}
    return __proxy__['panos.call'](query)

def get_admins_active():
    if False:
        i = 10
        return i + 15
    "\n    Show active administrators.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_admins_active\n\n    "
    query = {'type': 'op', 'cmd': '<show><admins></admins></show>'}
    return __proxy__['panos.call'](query)

def get_admins_all():
    if False:
        for i in range(10):
            print('nop')
    "\n    Show all administrators.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_admins_all\n\n    "
    query = {'type': 'op', 'cmd': '<show><admins><all></all></admins></show>'}
    return __proxy__['panos.call'](query)

def get_antivirus_info():
    if False:
        while True:
            i = 10
    "\n    Show information about available anti-virus packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_antivirus_info\n\n    "
    query = {'type': 'op', 'cmd': '<request><anti-virus><upgrade><info></info></upgrade></anti-virus></request>'}
    return __proxy__['panos.call'](query)

def get_arp():
    if False:
        print('Hello World!')
    "\n    Show ARP information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_arp\n\n    "
    query = {'type': 'op', 'cmd': "<show><arp><entry name = 'all'/></arp></show>"}
    return __proxy__['panos.call'](query)

def get_cli_idle_timeout():
    if False:
        i = 10
        return i + 15
    "\n    Show timeout information for this administrative session.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_cli_idle_timeout\n\n    "
    query = {'type': 'op', 'cmd': '<show><cli><idle-timeout></idle-timeout></cli></show>'}
    return __proxy__['panos.call'](query)

def get_cli_permissions():
    if False:
        i = 10
        return i + 15
    "\n    Show cli administrative permissions.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_cli_permissions\n\n    "
    query = {'type': 'op', 'cmd': '<show><cli><permissions></permissions></cli></show>'}
    return __proxy__['panos.call'](query)

def get_disk_usage():
    if False:
        for i in range(10):
            print('nop')
    "\n    Report filesystem disk space usage.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_disk_usage\n\n    "
    query = {'type': 'op', 'cmd': '<show><system><disk-space></disk-space></system></show>'}
    return __proxy__['panos.call'](query)

def get_dns_server_config():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the DNS server configuration from the candidate configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_dns_server_config\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/dns-setting/servers"}
    return __proxy__['panos.call'](query)

def get_domain_config():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the domain name configuration from the candidate configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_domain_config\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/domain"}
    return __proxy__['panos.call'](query)

def get_dos_blocks():
    if False:
        return 10
    "\n    Show the DoS block-ip table.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_dos_blocks\n\n    "
    query = {'type': 'op', 'cmd': '<show><dos-block-table><all></all></dos-block-table></show>'}
    return __proxy__['panos.call'](query)

def get_fqdn_cache():
    if False:
        for i in range(10):
            print('nop')
    "\n    Print FQDNs used in rules and their IPs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_fqdn_cache\n\n    "
    query = {'type': 'op', 'cmd': '<request><system><fqdn><show></show></fqdn></system></request>'}
    return __proxy__['panos.call'](query)

def get_ha_config():
    if False:
        return 10
    "\n    Get the high availability configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_ha_config\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/high-availability"}
    return __proxy__['panos.call'](query)

def get_ha_link():
    if False:
        for i in range(10):
            print('nop')
    "\n     Show high-availability link-monitoring state.\n\n     CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' panos.get_ha_link\n\n    "
    query = {'type': 'op', 'cmd': '<show><high-availability><link-monitoring></link-monitoring></high-availability></show>'}
    return __proxy__['panos.call'](query)

def get_ha_path():
    if False:
        return 10
    "\n    Show high-availability path-monitoring state.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_ha_path\n\n    "
    query = {'type': 'op', 'cmd': '<show><high-availability><path-monitoring></path-monitoring></high-availability></show>'}
    return __proxy__['panos.call'](query)

def get_ha_state():
    if False:
        for i in range(10):
            print('nop')
    "\n    Show high-availability state information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_ha_state\n\n    "
    query = {'type': 'op', 'cmd': '<show><high-availability><state></state></high-availability></show>'}
    return __proxy__['panos.call'](query)

def get_ha_transitions():
    if False:
        i = 10
        return i + 15
    "\n    Show high-availability transition statistic information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_ha_transitions\n\n    "
    query = {'type': 'op', 'cmd': '<show><high-availability><transitions></transitions></high-availability></show>'}
    return __proxy__['panos.call'](query)

def get_hostname():
    if False:
        while True:
            i = 10
    "\n    Get the hostname of the device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_hostname\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/hostname"}
    return __proxy__['panos.call'](query)

def get_interface_counters(name='all'):
    if False:
        print('Hello World!')
    "\n    Get the counter statistics for interfaces.\n\n    Args:\n        name (str): The name of the interface to view. By default, all interface statistics are viewed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_interface_counters\n        salt '*' panos.get_interface_counters ethernet1/1\n\n    "
    query = {'type': 'op', 'cmd': '<show><counter><interface>{}</interface></counter></show>'.format(name)}
    return __proxy__['panos.call'](query)

def get_interfaces(name='all'):
    if False:
        return 10
    "\n    Show interface information.\n\n    Args:\n        name (str): The name of the interface to view. By default, all interface statistics are viewed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_interfaces\n        salt '*' panos.get_interfaces ethernet1/1\n\n    "
    query = {'type': 'op', 'cmd': '<show><interface>{}</interface></show>'.format(name)}
    return __proxy__['panos.call'](query)

def get_job(jid=None):
    if False:
        while True:
            i = 10
    "\n    List all a single job by ID.\n\n    jid\n        The ID of the job to retrieve.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_job jid=15\n\n    "
    if not jid:
        raise CommandExecutionError('ID option must not be none.')
    query = {'type': 'op', 'cmd': '<show><jobs><id>{}</id></jobs></show>'.format(jid)}
    return __proxy__['panos.call'](query)

def get_jobs(state='all'):
    if False:
        return 10
    "\n    List all jobs on the device.\n\n    state\n        The state of the jobs to display. Valid options are all, pending, or processed. Pending jobs are jobs\n        that are currently in a running or waiting state. Processed jobs are jobs that have completed\n        execution.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_jobs\n        salt '*' panos.get_jobs state=pending\n\n    "
    if state.lower() == 'all':
        query = {'type': 'op', 'cmd': '<show><jobs><all></all></jobs></show>'}
    elif state.lower() == 'pending':
        query = {'type': 'op', 'cmd': '<show><jobs><pending></pending></jobs></show>'}
    elif state.lower() == 'processed':
        query = {'type': 'op', 'cmd': '<show><jobs><processed></processed></jobs></show>'}
    else:
        raise CommandExecutionError('The state parameter must be all, pending, or processed.')
    return __proxy__['panos.call'](query)

def get_lacp():
    if False:
        return 10
    "\n    Show LACP state.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_lacp\n\n    "
    query = {'type': 'op', 'cmd': '<show><lacp><aggregate-ethernet>all</aggregate-ethernet></lacp></show>'}
    return __proxy__['panos.call'](query)

def get_license_info():
    if False:
        i = 10
        return i + 15
    "\n    Show information about owned license(s).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_license_info\n\n    "
    query = {'type': 'op', 'cmd': '<request><license><info></info></license></request>'}
    return __proxy__['panos.call'](query)

def get_license_tokens():
    if False:
        return 10
    "\n    Show license token files for manual license deactivation.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_license_tokens\n\n    "
    query = {'type': 'op', 'cmd': '<show><license-token-files></license-token-files></show>'}
    return __proxy__['panos.call'](query)

def get_lldp_config():
    if False:
        print('Hello World!')
    "\n    Show lldp config for interfaces.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_lldp_config\n\n    "
    query = {'type': 'op', 'cmd': '<show><lldp><config>all</config></lldp></show>'}
    return __proxy__['panos.call'](query)

def get_lldp_counters():
    if False:
        i = 10
        return i + 15
    "\n    Show lldp counters for interfaces.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_lldp_counters\n\n    "
    query = {'type': 'op', 'cmd': '<show><lldp><counters>all</counters></lldp></show>'}
    return __proxy__['panos.call'](query)

def get_lldp_local():
    if False:
        print('Hello World!')
    "\n    Show lldp local info for interfaces.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_lldp_local\n\n    "
    query = {'type': 'op', 'cmd': '<show><lldp><local>all</local></lldp></show>'}
    return __proxy__['panos.call'](query)

def get_lldp_neighbors():
    if False:
        return 10
    "\n    Show lldp neighbors info for interfaces.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_lldp_neighbors\n\n    "
    query = {'type': 'op', 'cmd': '<show><lldp><neighbors>all</neighbors></lldp></show>'}
    return __proxy__['panos.call'](query)

def get_local_admins():
    if False:
        for i in range(10):
            print('nop')
    "\n    Show all local administrator accounts.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_local_admins\n\n    "
    admin_list = get_users_config()
    response = []
    if 'users' not in admin_list['result']:
        return response
    if isinstance(admin_list['result']['users']['entry'], list):
        for entry in admin_list['result']['users']['entry']:
            response.append(entry['name'])
    else:
        response.append(admin_list['result']['users']['entry']['name'])
    return response

def get_logdb_quota():
    if False:
        print('Hello World!')
    "\n    Report the logdb quotas.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_logdb_quota\n\n    "
    query = {'type': 'op', 'cmd': '<show><system><logdb-quota></logdb-quota></system></show>'}
    return __proxy__['panos.call'](query)

def get_master_key():
    if False:
        print('Hello World!')
    "\n    Get the master key properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_master_key\n\n    "
    query = {'type': 'op', 'cmd': '<show><system><masterkey-properties></masterkey-properties></system></show>'}
    return __proxy__['panos.call'](query)

def get_ntp_config():
    if False:
        while True:
            i = 10
    "\n    Get the NTP configuration from the candidate configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_ntp_config\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers"}
    return __proxy__['panos.call'](query)

def get_ntp_servers():
    if False:
        return 10
    "\n    Get list of configured NTP servers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_ntp_servers\n\n    "
    query = {'type': 'op', 'cmd': '<show><ntp></ntp></show>'}
    return __proxy__['panos.call'](query)

def get_operational_mode():
    if False:
        while True:
            i = 10
    "\n    Show device operational mode setting.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_operational_mode\n\n    "
    query = {'type': 'op', 'cmd': '<show><operational-mode></operational-mode></show>'}
    return __proxy__['panos.call'](query)

def get_panorama_status():
    if False:
        i = 10
        return i + 15
    "\n    Show panorama connection status.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_panorama_status\n\n    "
    query = {'type': 'op', 'cmd': '<show><panorama-status></panorama-status></show>'}
    return __proxy__['panos.call'](query)

def get_permitted_ips():
    if False:
        i = 10
        return i + 15
    "\n    Get the IP addresses that are permitted to establish management connections to the device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_permitted_ips\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/permitted-ip"}
    return __proxy__['panos.call'](query)

def get_platform():
    if False:
        print('Hello World!')
    "\n    Get the platform model information and limitations.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_platform\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/platform"}
    return __proxy__['panos.call'](query)

def get_predefined_application(application=None):
    if False:
        while True:
            i = 10
    "\n    Get the configuration for the specified pre-defined application object. This will only return pre-defined\n    application objects.\n\n    application(str): The name of the pre-defined application object.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_predefined_application saltstack\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/predefined/application/entry[@name='{}']".format(application)}
    return __proxy__['panos.call'](query)

def get_security_rule(rulename=None, vsys='1'):
    if False:
        i = 10
        return i + 15
    "\n    Get the candidate configuration for the specified security rule.\n\n    rulename(str): The name of the security rule.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_security_rule rule01\n        salt '*' panos.get_security_rule rule01 3\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/rulebase/security/rules/entry[@name='{}']".format(vsys, rulename)}
    return __proxy__['panos.call'](query)

def get_service(service=None, vsys='1'):
    if False:
        i = 10
        return i + 15
    "\n    Get the candidate configuration for the specified service object. This will not return services that are marked\n    as pre-defined objects.\n\n    service(str): The name of the service object.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_service tcp-443\n        salt '*' panos.get_service tcp-443 3\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/service/entry[@name='{}']".format(vsys, service)}
    return __proxy__['panos.call'](query)

def get_service_group(servicegroup=None, vsys='1'):
    if False:
        print('Hello World!')
    "\n    Get the candidate configuration for the specified service group. This will not return service groups that are\n    marked as pre-defined objects.\n\n    servicegroup(str): The name of the service group.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_service_group foobar\n        salt '*' panos.get_service_group foobar 3\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/service-group/entry[@name='{}']".format(vsys, servicegroup)}
    return __proxy__['panos.call'](query)

def get_session_info():
    if False:
        while True:
            i = 10
    "\n    Show device session statistics.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_session_info\n\n    "
    query = {'type': 'op', 'cmd': '<show><session><info></info></session></show>'}
    return __proxy__['panos.call'](query)

def get_snmp_config():
    if False:
        return 10
    "\n    Get the SNMP configuration from the device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_snmp_config\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/snmp-setting"}
    return __proxy__['panos.call'](query)

def get_software_info():
    if False:
        print('Hello World!')
    "\n    Show information about available software packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_software_info\n\n    "
    query = {'type': 'op', 'cmd': '<request><system><software><info></info></software></system></request>'}
    return __proxy__['panos.call'](query)

def get_system_date_time():
    if False:
        return 10
    "\n    Get the system date/time.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_system_date_time\n\n    "
    query = {'type': 'op', 'cmd': '<show><clock></clock></show>'}
    return __proxy__['panos.call'](query)

def get_system_files():
    if False:
        while True:
            i = 10
    "\n    List important files in the system.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_system_files\n\n    "
    query = {'type': 'op', 'cmd': '<show><system><files></files></system></show>'}
    return __proxy__['panos.call'](query)

def get_system_info():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the system information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_system_info\n\n    "
    query = {'type': 'op', 'cmd': '<show><system><info></info></system></show>'}
    return __proxy__['panos.call'](query)

def get_system_services():
    if False:
        for i in range(10):
            print('nop')
    "\n    Show system services.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_system_services\n\n    "
    query = {'type': 'op', 'cmd': '<show><system><services></services></system></show>'}
    return __proxy__['panos.call'](query)

def get_system_state(mask=None):
    if False:
        return 10
    "\n    Show the system state variables.\n\n    mask\n        Filters by a subtree or a wildcard.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_system_state\n        salt '*' panos.get_system_state mask=cfg.ha.config.enabled\n        salt '*' panos.get_system_state mask=cfg.ha.*\n\n    "
    if mask:
        query = {'type': 'op', 'cmd': '<show><system><state><filter>{}</filter></state></system></show>'.format(mask)}
    else:
        query = {'type': 'op', 'cmd': '<show><system><state></state></system></show>'}
    return __proxy__['panos.call'](query)

def get_uncommitted_changes():
    if False:
        i = 10
        return i + 15
    "\n    Retrieve a list of all uncommitted changes on the device.\n    Requires PANOS version 8.0.0 or greater.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_uncommitted_changes\n\n    "
    _required_version = '8.0.0'
    if not __proxy__['panos.is_required_version'](_required_version):
        return (False, 'The panos device requires version {} or greater for this command.'.format(_required_version))
    query = {'type': 'op', 'cmd': '<show><config><list><changes></changes></list></config></show>'}
    return __proxy__['panos.call'](query)

def get_users_config():
    if False:
        while True:
            i = 10
    "\n    Get the local administrative user account configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_users_config\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': '/config/mgt-config/users'}
    return __proxy__['panos.call'](query)

def get_vlans():
    if False:
        i = 10
        return i + 15
    "\n    Show all VLAN information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_vlans\n\n    "
    query = {'type': 'op', 'cmd': '<show><vlan>all</vlan></show>'}
    return __proxy__['panos.call'](query)

def get_xpath(xpath=''):
    if False:
        return 10
    "\n    Retrieve a specified xpath from the candidate configuration.\n\n    xpath(str): The specified xpath in the candidate configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_xpath /config/shared/service\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': xpath}
    return __proxy__['panos.call'](query)

def get_zone(zone='', vsys='1'):
    if False:
        i = 10
        return i + 15
    "\n    Get the candidate configuration for the specified zone.\n\n    zone(str): The name of the zone.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_zone trust\n        salt '*' panos.get_zone trust 2\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/zone/entry[@name='{}']".format(vsys, zone)}
    return __proxy__['panos.call'](query)

def get_zones(vsys='1'):
    if False:
        print('Hello World!')
    "\n    Get all the zones in the candidate configuration.\n\n    vsys(str): The string representation of the VSYS ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.get_zones\n        salt '*' panos.get_zones 2\n\n    "
    query = {'type': 'config', 'action': 'get', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys{}']/zone".format(vsys)}
    return __proxy__['panos.call'](query)

def install_antivirus(version=None, latest=False, synch=False, skip_commit=False):
    if False:
        i = 10
        return i + 15
    "\n    Install anti-virus packages.\n\n    Args:\n        version(str): The version of the PANOS file to install.\n\n        latest(bool): If true, the latest anti-virus file will be installed.\n                      The specified version option will be ignored.\n\n        synch(bool): If true, the anti-virus will synch to the peer unit.\n\n        skip_commit(bool): If true, the install will skip committing to the device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.install_antivirus 8.0.0\n\n    "
    if not version and latest is False:
        raise CommandExecutionError('Version option must not be none.')
    if synch is True:
        s = 'yes'
    else:
        s = 'no'
    if skip_commit is True:
        c = 'yes'
    else:
        c = 'no'
    if latest is True:
        query = {'type': 'op', 'cmd': '<request><anti-virus><upgrade><install><commit>{}</commit><sync-to-peer>{}</sync-to-peer><version>latest</version></install></upgrade></anti-virus></request>'.format(c, s)}
    else:
        query = {'type': 'op', 'cmd': '<request><anti-virus><upgrade><install><commit>{}</commit><sync-to-peer>{}</sync-to-peer><version>{}</version></install></upgrade></anti-virus></request>'.format(c, s, version)}
    return _get_job_results(query)

def install_license():
    if False:
        i = 10
        return i + 15
    "\n    Install the license key(s).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.install_license\n\n    "
    query = {'type': 'op', 'cmd': '<request><license><install></install></license></request>'}
    return __proxy__['panos.call'](query)

def install_software(version=None):
    if False:
        return 10
    "\n    Upgrade to a software package by version.\n\n    Args:\n        version(str): The version of the PANOS file to install.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.install_license 8.0.0\n\n    "
    if not version:
        raise CommandExecutionError('Version option must not be none.')
    query = {'type': 'op', 'cmd': '<request><system><software><install><version>{}</version></install></software></system></request>'.format(version)}
    return _get_job_results(query)

def reboot():
    if False:
        return 10
    "\n    Reboot a running system.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.reboot\n\n    "
    query = {'type': 'op', 'cmd': '<request><restart><system></system></restart></request>'}
    return __proxy__['panos.call'](query)

def refresh_fqdn_cache(force=False):
    if False:
        while True:
            i = 10
    "\n    Force refreshes all FQDNs used in rules.\n\n    force\n        Forces all fqdn refresh\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.refresh_fqdn_cache\n        salt '*' panos.refresh_fqdn_cache force=True\n\n    "
    if not isinstance(force, bool):
        raise CommandExecutionError('Force option must be boolean.')
    if force:
        query = {'type': 'op', 'cmd': '<request><system><fqdn><refresh><force>yes</force></refresh></fqdn></system></request>'}
    else:
        query = {'type': 'op', 'cmd': '<request><system><fqdn><refresh></refresh></fqdn></system></request>'}
    return __proxy__['panos.call'](query)

def remove_config_lock():
    if False:
        i = 10
        return i + 15
    "\n    Release config lock previously held.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.remove_config_lock\n\n    "
    query = {'type': 'op', 'cmd': '<request><config-lock><remove></remove></config-lock></request>'}
    return __proxy__['panos.call'](query)

def resolve_address(address=None, vsys=None):
    if False:
        while True:
            i = 10
    "\n    Resolve address to ip address.\n    Required version 7.0.0 or greater.\n\n    address\n        Address name you want to resolve.\n\n    vsys\n        The vsys name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.resolve_address foo.bar.com\n        salt '*' panos.resolve_address foo.bar.com vsys=2\n\n    "
    _required_version = '7.0.0'
    if not __proxy__['panos.is_required_version'](_required_version):
        return (False, 'The panos device requires version {} or greater for this command.'.format(_required_version))
    if not address:
        raise CommandExecutionError('FQDN to resolve must be provided as address.')
    if not vsys:
        query = {'type': 'op', 'cmd': '<request><resolve><address>{}</address></resolve></request>'.format(address)}
    else:
        query = {'type': 'op', 'cmd': '<request><resolve><vsys>{}</vsys><address>{}</address></resolve></request>'.format(vsys, address)}
    return __proxy__['panos.call'](query)

def save_device_config(filename=None):
    if False:
        print('Hello World!')
    "\n    Save device configuration to a named file.\n\n    filename\n        The filename to save the configuration to.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.save_device_config foo.xml\n\n    "
    if not filename:
        raise CommandExecutionError('Filename must not be empty.')
    query = {'type': 'op', 'cmd': '<save><config><to>{}</to></config></save>'.format(filename)}
    return __proxy__['panos.call'](query)

def save_device_state():
    if False:
        return 10
    "\n    Save files needed to restore device to local disk.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.save_device_state\n\n    "
    query = {'type': 'op', 'cmd': '<save><device-state></device-state></save>'}
    return __proxy__['panos.call'](query)

def set_authentication_profile(profile=None, deploy=False):
    if False:
        print('Hello World!')
    "\n    Set the authentication profile of the Palo Alto proxy minion. A commit will be required before this is processed.\n\n    CLI Example:\n\n    Args:\n        profile (str): The name of the authentication profile to set.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_authentication_profile foo\n        salt '*' panos.set_authentication_profile foo deploy=True\n\n    "
    if not profile:
        raise CommandExecutionError('Profile name option must not be none.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/authentication-profile", 'element': '<authentication-profile>{}</authentication-profile>'.format(profile)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_hostname(hostname=None, deploy=False):
    if False:
        i = 10
        return i + 15
    "\n    Set the hostname of the Palo Alto proxy minion. A commit will be required before this is processed.\n\n    CLI Example:\n\n    Args:\n        hostname (str): The hostname to set\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_hostname newhostname\n        salt '*' panos.set_hostname newhostname deploy=True\n\n    "
    if not hostname:
        raise CommandExecutionError('Hostname option must not be none.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system", 'element': '<hostname>{}</hostname>'.format(hostname)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_icmp(enabled=True, deploy=False):
    if False:
        while True:
            i = 10
    "\n    Enables or disables the ICMP management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_icmp\n        salt '*' panos.set_management_icmp enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-icmp>{}</disable-icmp>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_http(enabled=True, deploy=False):
    if False:
        print('Hello World!')
    "\n    Enables or disables the HTTP management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_http\n        salt '*' panos.set_management_http enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-http>{}</disable-http>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_https(enabled=True, deploy=False):
    if False:
        i = 10
        return i + 15
    "\n    Enables or disables the HTTPS management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_https\n        salt '*' panos.set_management_https enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-https>{}</disable-https>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_ocsp(enabled=True, deploy=False):
    if False:
        while True:
            i = 10
    "\n    Enables or disables the HTTP OCSP management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_ocsp\n        salt '*' panos.set_management_ocsp enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-http-ocsp>{}</disable-http-ocsp>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_snmp(enabled=True, deploy=False):
    if False:
        return 10
    "\n    Enables or disables the SNMP management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_snmp\n        salt '*' panos.set_management_snmp enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-snmp>{}</disable-snmp>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_ssh(enabled=True, deploy=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Enables or disables the SSH management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_ssh\n        salt '*' panos.set_management_ssh enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-ssh>{}</disable-ssh>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_management_telnet(enabled=True, deploy=False):
    if False:
        return 10
    "\n    Enables or disables the Telnet management service on the device.\n\n    CLI Example:\n\n    Args:\n        enabled (bool): If true the service will be enabled. If false the service will be disabled.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_management_telnet\n        salt '*' panos.set_management_telnet enabled=False deploy=True\n\n    "
    if enabled is True:
        value = 'no'
    elif enabled is False:
        value = 'yes'
    else:
        raise CommandExecutionError('Invalid option provided for service enabled option.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/service", 'element': '<disable-telnet>{}</disable-telnet>'.format(value)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_ntp_authentication(target=None, authentication_type=None, key_id=None, authentication_key=None, algorithm=None, deploy=False):
    if False:
        return 10
    "\n    Set the NTP authentication of the Palo Alto proxy minion. A commit will be required before this is processed.\n\n    CLI Example:\n\n    Args:\n        target(str): Determines the target of the authentication. Valid options are primary, secondary, or both.\n\n        authentication_type(str): The authentication type to be used. Valid options are symmetric, autokey, and none.\n\n        key_id(int): The NTP authentication key ID.\n\n        authentication_key(str): The authentication key.\n\n        algorithm(str): The algorithm type to be used for a symmetric key. Valid options are md5 and sha1.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' ntp.set_authentication target=both authentication_type=autokey\n        salt '*' ntp.set_authentication target=primary authentication_type=none\n        salt '*' ntp.set_authentication target=both authentication_type=symmetric key_id=15 authentication_key=mykey algorithm=md5\n        salt '*' ntp.set_authentication target=both authentication_type=symmetric key_id=15 authentication_key=mykey algorithm=md5 deploy=True\n\n    "
    ret = {}
    if target not in ['primary', 'secondary', 'both']:
        raise salt.exceptions.CommandExecutionError('Target option must be primary, secondary, or both.')
    if authentication_type not in ['symmetric', 'autokey', 'none']:
        raise salt.exceptions.CommandExecutionError('Type option must be symmetric, autokey, or both.')
    if authentication_type == 'symmetric' and (not authentication_key):
        raise salt.exceptions.CommandExecutionError('When using symmetric authentication, authentication_key must be provided.')
    if authentication_type == 'symmetric' and (not key_id):
        raise salt.exceptions.CommandExecutionError('When using symmetric authentication, key_id must be provided.')
    if authentication_type == 'symmetric' and algorithm not in ['md5', 'sha1']:
        raise salt.exceptions.CommandExecutionError('When using symmetric authentication, algorithm must be md5 or sha1.')
    if authentication_type == 'symmetric':
        if target == 'primary' or target == 'both':
            query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/primary-ntp-server/authentication-type", 'element': '<symmetric-key><algorithm><{0}><authentication-key>{1}</authentication-key></{0}></algorithm><key-id>{2}</key-id></symmetric-key>'.format(algorithm, authentication_key, key_id)}
            ret.update({'primary_server': __proxy__['panos.call'](query)})
        if target == 'secondary' or target == 'both':
            query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/secondary-ntp-server/authentication-type", 'element': '<symmetric-key><algorithm><{0}><authentication-key>{1}</authentication-key></{0}></algorithm><key-id>{2}</key-id></symmetric-key>'.format(algorithm, authentication_key, key_id)}
            ret.update({'secondary_server': __proxy__['panos.call'](query)})
    elif authentication_type == 'autokey':
        if target == 'primary' or target == 'both':
            query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/primary-ntp-server/authentication-type", 'element': '<autokey/>'}
            ret.update({'primary_server': __proxy__['panos.call'](query)})
        if target == 'secondary' or target == 'both':
            query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/secondary-ntp-server/authentication-type", 'element': '<autokey/>'}
            ret.update({'secondary_server': __proxy__['panos.call'](query)})
    elif authentication_type == 'none':
        if target == 'primary' or target == 'both':
            query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/primary-ntp-server/authentication-type", 'element': '<none/>'}
            ret.update({'primary_server': __proxy__['panos.call'](query)})
        if target == 'secondary' or target == 'both':
            query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/secondary-ntp-server/authentication-type", 'element': '<none/>'}
            ret.update({'secondary_server': __proxy__['panos.call'](query)})
    if deploy is True:
        ret.update(commit())
    return ret

def set_ntp_servers(primary_server=None, secondary_server=None, deploy=False):
    if False:
        print('Hello World!')
    "\n    Set the NTP servers of the Palo Alto proxy minion. A commit will be required before this is processed.\n\n    CLI Example:\n\n    Args:\n        primary_server(str): The primary NTP server IP address or FQDN.\n\n        secondary_server(str): The secondary NTP server IP address or FQDN.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' ntp.set_servers 0.pool.ntp.org 1.pool.ntp.org\n        salt '*' ntp.set_servers primary_server=0.pool.ntp.org secondary_server=1.pool.ntp.org\n        salt '*' ntp.ser_servers 0.pool.ntp.org 1.pool.ntp.org deploy=True\n\n    "
    ret = {}
    if primary_server:
        query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/primary-ntp-server", 'element': '<ntp-server-address>{}</ntp-server-address>'.format(primary_server)}
        ret.update({'primary_server': __proxy__['panos.call'](query)})
    if secondary_server:
        query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/ntp-servers/secondary-ntp-server", 'element': '<ntp-server-address>{}</ntp-server-address>'.format(secondary_server)}
        ret.update({'secondary_server': __proxy__['panos.call'](query)})
    if deploy is True:
        ret.update(commit())
    return ret

def set_permitted_ip(address=None, deploy=False):
    if False:
        i = 10
        return i + 15
    "\n    Add an IPv4 address or network to the permitted IP list.\n\n    CLI Example:\n\n    Args:\n        address (str): The IPv4 address or network to allow access to add to the Palo Alto device.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_permitted_ip 10.0.0.1\n        salt '*' panos.set_permitted_ip 10.0.0.0/24\n        salt '*' panos.set_permitted_ip 10.0.0.1 deploy=True\n\n    "
    if not address:
        raise CommandExecutionError('Address option must not be empty.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/permitted-ip", 'element': "<entry name='{}'></entry>".format(address)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def set_timezone(tz=None, deploy=False):
    if False:
        i = 10
        return i + 15
    "\n    Set the timezone of the Palo Alto proxy minion. A commit will be required before this is processed.\n\n    CLI Example:\n\n    Args:\n        tz (str): The name of the timezone to set.\n\n        deploy (bool): If true then commit the full candidate configuration, if false only set pending change.\n\n    .. code-block:: bash\n\n        salt '*' panos.set_timezone UTC\n        salt '*' panos.set_timezone UTC deploy=True\n\n    "
    if not tz:
        raise CommandExecutionError('Timezone name option must not be none.')
    ret = {}
    query = {'type': 'config', 'action': 'set', 'xpath': "/config/devices/entry[@name='localhost.localdomain']/deviceconfig/system/timezone", 'element': '<timezone>{}</timezone>'.format(tz)}
    ret.update(__proxy__['panos.call'](query))
    if deploy is True:
        ret.update(commit())
    return ret

def shutdown():
    if False:
        i = 10
        return i + 15
    "\n    Shutdown a running system.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.shutdown\n\n    "
    query = {'type': 'op', 'cmd': '<request><shutdown><system></system></shutdown></request>'}
    return __proxy__['panos.call'](query)

def test_fib_route(ip=None, vr='vr1'):
    if False:
        return 10
    "\n    Perform a route lookup within active route table (fib).\n\n    ip (str): The destination IP address to test.\n\n    vr (str): The name of the virtual router to test.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.test_fib_route 4.2.2.2\n        salt '*' panos.test_fib_route 4.2.2.2 my-vr\n\n    "
    xpath = '<test><routing><fib-lookup>'
    if ip:
        xpath += '<ip>{}</ip>'.format(ip)
    if vr:
        xpath += '<virtual-router>{}</virtual-router>'.format(vr)
    xpath += '</fib-lookup></routing></test>'
    query = {'type': 'op', 'cmd': xpath}
    return __proxy__['panos.call'](query)

def test_security_policy(sourcezone=None, destinationzone=None, source=None, destination=None, protocol=None, port=None, application=None, category=None, vsys='1', allrules=False):
    if False:
        return 10
    "\n    Checks which security policy as connection will match on the device.\n\n    sourcezone (str): The source zone matched against the connection.\n\n    destinationzone (str): The destination zone matched against the connection.\n\n    source (str): The source address. This must be a single IP address.\n\n    destination (str): The destination address. This must be a single IP address.\n\n    protocol (int): The protocol number for the connection. This is the numerical representation of the protocol.\n\n    port (int): The port number for the connection.\n\n    application (str): The application that should be matched.\n\n    category (str): The category that should be matched.\n\n    vsys (int): The numerical representation of the VSYS ID.\n\n    allrules (bool): Show all potential match rules until first allow rule.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.test_security_policy sourcezone=trust destinationzone=untrust protocol=6 port=22\n        salt '*' panos.test_security_policy sourcezone=trust destinationzone=untrust protocol=6 port=22 vsys=2\n\n    "
    xpath = '<test><security-policy-match>'
    if sourcezone:
        xpath += '<from>{}</from>'.format(sourcezone)
    if destinationzone:
        xpath += '<to>{}</to>'.format(destinationzone)
    if source:
        xpath += '<source>{}</source>'.format(source)
    if destination:
        xpath += '<destination>{}</destination>'.format(destination)
    if protocol:
        xpath += '<protocol>{}</protocol>'.format(protocol)
    if port:
        xpath += '<destination-port>{}</destination-port>'.format(port)
    if application:
        xpath += '<application>{}</application>'.format(application)
    if category:
        xpath += '<category>{}</category>'.format(category)
    if allrules:
        xpath += '<show-all>yes</show-all>'
    xpath += '</security-policy-match></test>'
    query = {'type': 'op', 'vsys': 'vsys{}'.format(vsys), 'cmd': xpath}
    return __proxy__['panos.call'](query)

def unlock_admin(username=None):
    if False:
        while True:
            i = 10
    "\n    Unlocks a locked administrator account.\n\n    username\n        Username of the administrator.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' panos.unlock_admin username=bob\n\n    "
    if not username:
        raise CommandExecutionError('Username option must not be none.')
    query = {'type': 'op', 'cmd': '<set><management-server><unlock><admin>{}</admin></unlock></management-server></set>'.format(username)}
    return __proxy__['panos.call'](query)