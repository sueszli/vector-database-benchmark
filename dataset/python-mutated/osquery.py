"""
Support for OSQuery - https://osquery.io.

.. versionadded:: 2015.8.0
"""
import logging
import salt.utils.json
import salt.utils.path
import salt.utils.platform
log = logging.getLogger(__name__)
__func_alias__ = {'file_': 'file', 'hash_': 'hash', 'time_': 'time'}
__virtualname__ = 'osquery'

def __virtual__():
    if False:
        return 10
    if salt.utils.path.which('osqueryi'):
        return __virtualname__
    return (False, 'The osquery execution module cannot be loaded: osqueryi binary is not in the path.')

def _table_attrs(table):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to find valid table attributes\n    '
    cmd = ['osqueryi'] + ['--json'] + ['pragma table_info({})'.format(table)]
    res = __salt__['cmd.run_all'](cmd)
    if res['retcode'] == 0:
        attrs = []
        text = salt.utils.json.loads(res['stdout'])
        for item in text:
            attrs.append(item['name'])
        return attrs
    return False

def _osquery(sql, format='json'):
    if False:
        while True:
            i = 10
    '\n    Helper function to run raw osquery queries\n    '
    ret = {'result': True}
    cmd = ['osqueryi'] + ['--json'] + [sql]
    res = __salt__['cmd.run_all'](cmd)
    if res['stderr']:
        ret['result'] = False
        ret['error'] = res['stderr']
    else:
        ret['data'] = salt.utils.json.loads(res['stdout'])
    log.debug('== %s ==', ret)
    return ret

def _osquery_cmd(table, attrs=None, where=None, format='json'):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to run osquery queries\n    '
    ret = {'result': True}
    if attrs:
        if isinstance(attrs, list):
            valid_attrs = _table_attrs(table)
            if valid_attrs:
                for a in attrs:
                    if a not in valid_attrs:
                        ret['result'] = False
                        ret['comment'] = '{} is not a valid attribute for table {}'.format(a, table)
                        return ret
                _attrs = ','.join(attrs)
            else:
                ret['result'] = False
                ret['comment'] = 'Invalid table {}.'.format(table)
                return ret
        else:
            ret['comment'] = 'attrs must be specified as a list.'
            ret['result'] = False
            return ret
    else:
        _attrs = '*'
    sql = 'select {} from {}'.format(_attrs, table)
    if where:
        sql = '{} where {}'.format(sql, where)
    sql = '{};'.format(sql)
    res = _osquery(sql)
    if res['result']:
        ret['data'] = res['data']
    else:
        ret['comment'] = res['error']
    return ret

def version():
    if False:
        return 10
    "\n    Return version of osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.version\n    "
    _false_return = {'result': False, 'comment': 'OSQuery version unavailable.'}
    res = _osquery_cmd(table='osquery_info', attrs=['version'])
    if 'result' in res and res['result']:
        if 'data' in res and isinstance(res['data'], list):
            return res['data'][0].get('version', '') or _false_return
    return _false_return

def rpm_packages(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return cpuid information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.rpm_packages\n    "
    if __grains__['os_family'] == 'RedHat':
        return _osquery_cmd(table='rpm_packages', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Red Hat based systems.'}

def kernel_integrity(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return kernel_integrity information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.kernel_integrity\n    "
    if __grains__['os_family'] in ['RedHat', 'Debian']:
        return _osquery_cmd(table='kernel_integrity', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Red Hat or Debian based systems.'}

def kernel_modules(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return kernel_modules information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.kernel_modules\n    "
    if __grains__['os_family'] in ['RedHat', 'Debian']:
        return _osquery_cmd(table='kernel_modules', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Red Hat or Debian based systems.'}

def memory_map(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return memory_map information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.memory_map\n    "
    if __grains__['os_family'] in ['RedHat', 'Debian']:
        return _osquery_cmd(table='memory_map', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Red Hat or Debian based systems.'}

def process_memory_map(attrs=None, where=None):
    if False:
        return 10
    "\n    Return process_memory_map information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.process_memory_map\n    "
    if __grains__['os_family'] in ['RedHat', 'Debian']:
        return _osquery_cmd(table='process_memory_map', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Red Hat or Debian based systems.'}

def shared_memory(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return shared_memory information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.shared_memory\n    "
    if __grains__['os_family'] in ['RedHat', 'Debian']:
        return _osquery_cmd(table='shared_memory', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Red Hat or Debian based systems.'}

def apt_sources(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return apt_sources information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.apt_sources\n    "
    if __grains__['os_family'] == 'Debian':
        return _osquery_cmd(table='apt_sources', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Debian based systems.'}

def deb_packages(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return deb_packages information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.deb_packages\n    "
    if __grains__['os_family'] == 'Debian':
        return _osquery_cmd(table='deb_packages', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on Debian based systems.'}

def acpi_tables(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return acpi_tables information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.acpi_tables\n    "
    return _osquery_cmd(table='acpi_tables', attrs=attrs, where=where)

def arp_cache(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return arp_cache information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.arp_cache\n    "
    return _osquery_cmd(table='arp_cache', attrs=attrs, where=where)

def block_devices(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return block_devices information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.block_devices\n    "
    return _osquery_cmd(table='block_devices', attrs=attrs, where=where)

def cpuid(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return cpuid information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.cpuid\n    "
    return _osquery_cmd(table='cpuid', attrs=attrs, where=where)

def crontab(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return crontab information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.crontab\n    "
    return _osquery_cmd(table='crontab', attrs=attrs, where=where)

def etc_hosts(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return etc_hosts information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.etc_hosts\n    "
    return _osquery_cmd(table='etc_hosts', attrs=attrs, where=where)

def etc_services(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return etc_services information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.etc_services\n    "
    return _osquery_cmd(table='etc_services', attrs=attrs, where=where)

def file_changes(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return file_changes information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.file_changes\n    "
    return _osquery_cmd(table='file_changes', attrs=attrs, where=where)

def groups(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return groups information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.groups\n    "
    return _osquery_cmd(table='groups', attrs=attrs, where=where)

def hardware_events(attrs=None, where=None):
    if False:
        return 10
    "\n    Return hardware_events information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.hardware_events\n    "
    return _osquery_cmd(table='hardware_events', attrs=attrs, where=where)

def interface_addresses(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return interface_addresses information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.interface_addresses\n    "
    return _osquery_cmd(table='interface_addresses', attrs=attrs, where=where)

def interface_details(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return interface_details information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.interface_details\n    "
    return _osquery_cmd(table='interface_details', attrs=attrs, where=where)

def kernel_info(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return kernel_info information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.kernel_info\n    "
    return _osquery_cmd(table='kernel_info', attrs=attrs, where=where)

def last(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return last information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.last\n    "
    return _osquery_cmd(table='last', attrs=attrs, where=where)

def listening_ports(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return listening_ports information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.listening_ports\n    "
    return _osquery_cmd(table='listening_ports', attrs=attrs, where=where)

def logged_in_users(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return logged_in_users information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.logged_in_users\n    "
    return _osquery_cmd(table='logged_in_users', attrs=attrs, where=where)

def mounts(attrs=None, where=None):
    if False:
        return 10
    "\n    Return mounts information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.mounts\n    "
    return _osquery_cmd(table='mounts', attrs=attrs, where=where)

def os_version(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return os_version information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.os_version\n    "
    return _osquery_cmd(table='os_version', attrs=attrs, where=where)

def passwd_changes(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return passwd_changes information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.passwd_changes\n    "
    return _osquery_cmd(table='passwd_changes', attrs=attrs, where=where)

def pci_devices(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return pci_devices information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.pci_devices\n    "
    return _osquery_cmd(table='pci_devices', attrs=attrs, where=where)

def process_envs(attrs=None, where=None):
    if False:
        return 10
    "\n    Return process_envs information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.process_envs\n    "
    return _osquery_cmd(table='process_envs', attrs=attrs, where=where)

def process_open_files(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return process_open_files information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.process_open_files\n    "
    return _osquery_cmd(table='process_open_files', attrs=attrs, where=where)

def process_open_sockets(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return process_open_sockets information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.process_open_sockets\n    "
    return _osquery_cmd(table='process_open_sockets', attrs=attrs, where=where)

def processes(attrs=None, where=None):
    if False:
        return 10
    "\n    Return processes information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.processes\n    "
    return _osquery_cmd(table='processes', attrs=attrs, where=where)

def routes(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return routes information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.routes\n    "
    return _osquery_cmd(table='routes', attrs=attrs, where=where)

def shell_history(attrs=None, where=None):
    if False:
        return 10
    "\n    Return shell_history information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.shell_history\n    "
    return _osquery_cmd(table='shell_history', attrs=attrs, where=where)

def smbios_tables(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return smbios_tables information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.smbios_tables\n    "
    return _osquery_cmd(table='smbios_tables', attrs=attrs, where=where)

def suid_bin(attrs=None, where=None):
    if False:
        return 10
    "\n    Return suid_bin information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.suid_bin\n    "
    return _osquery_cmd(table='suid_bin', attrs=attrs, where=where)

def system_controls(attrs=None, where=None):
    if False:
        return 10
    "\n    Return system_controls information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.system_controls\n    "
    return _osquery_cmd(table='system_controls', attrs=attrs, where=where)

def usb_devices(attrs=None, where=None):
    if False:
        return 10
    "\n    Return usb_devices information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.usb_devices\n    "
    return _osquery_cmd(table='usb_devices', attrs=attrs, where=where)

def users(attrs=None, where=None):
    if False:
        return 10
    "\n    Return users information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.users\n    "
    return _osquery_cmd(table='users', attrs=attrs, where=where)

def alf(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return alf information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.alf\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='alf', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def alf_exceptions(attrs=None, where=None):
    if False:
        return 10
    "\n    Return alf_exceptions information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.alf_exceptions\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='alf_exceptions', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def alf_explicit_auths(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return alf_explicit_auths information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.alf_explicit_auths\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='alf_explicit_auths', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def alf_services(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return alf_services information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.alf_services\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='alf_services', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def apps(attrs=None, where=None):
    if False:
        return 10
    "\n    Return apps information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.apps\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='apps', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def certificates(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return certificates information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.certificates\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='certificates', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def chrome_extensions(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return chrome_extensions information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.chrome_extensions\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='chrome_extensions', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def firefox_addons(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return firefox_addons information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.firefox_addons\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='firefox_addons', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def homebrew_packages(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return homebrew_packages information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.homebrew_packages\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='homebrew_packages', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def iokit_devicetree(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return iokit_devicetree information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.iokit_devicetree\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='iokit_devicetree', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def iokit_registry(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return iokit_registry information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.iokit_registry\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='iokit_registry', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def kernel_extensions(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return kernel_extensions information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.kernel_extensions\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='kernel_extensions', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def keychain_items(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return keychain_items information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.keychain_items\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='keychain_items', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def launchd(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return launchd information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.launchd\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='launchd', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def nfs_shares(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return nfs_shares information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.nfs_shares\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='nfs_shares', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def nvram(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return nvram information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.nvram\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='nvram', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def preferences(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return preferences information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.preferences\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='preferences', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def quarantine(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return quarantine information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.quarantine\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='quarantine', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def safari_extensions(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return safari_extensions information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.safari_extensions\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='safari_extensions', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def startup_items(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return startup_items information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.startup_items\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='startup_items', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def xattr_where_from(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return xattr_where_from information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.xattr_where_from\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='xattr_where_from', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def xprotect_entries(attrs=None, where=None):
    if False:
        print('Hello World!')
    "\n    Return xprotect_entries information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.xprotect_entries\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='xprotect_entries', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def xprotect_reports(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return xprotect_reports information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.xprotect_reports\n    "
    if salt.utils.platform.is_darwin():
        return _osquery_cmd(table='xprotect_reports', attrs=attrs, where=where)
    return {'result': False, 'comment': 'Only available on macOS systems.'}

def file_(attrs=None, where=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return file information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.file\n    "
    return _osquery_cmd(table='file', attrs=attrs, where=where)

def hash_(attrs=None, where=None):
    if False:
        while True:
            i = 10
    "\n    Return hash information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.hash\n    "
    return _osquery_cmd(table='hash', attrs=attrs, where=where)

def osquery_extensions(attrs=None, where=None):
    if False:
        return 10
    "\n    Return osquery_extensions information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.osquery_extensions\n    "
    return _osquery_cmd(table='osquery_extensions', attrs=attrs, where=where)

def osquery_flags(attrs=None, where=None):
    if False:
        i = 10
        return i + 15
    "\n    Return osquery_flags information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.osquery_flags\n    "
    return _osquery_cmd(table='osquery_flags', attrs=attrs, where=where)

def osquery_info(attrs=None, where=None):
    if False:
        return 10
    "\n    Return osquery_info information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.osquery_info\n    "
    return _osquery_cmd(table='osquery_info', attrs=attrs, where=where)

def osquery_registry(attrs=None, where=None):
    if False:
        return 10
    "\n    Return osquery_registry information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.osquery_registry\n    "
    return _osquery_cmd(table='osquery_registry', attrs=attrs, where=where)

def time_(attrs=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return time information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' osquery.time\n    "
    return _osquery_cmd(table='time', attrs=attrs)

def query(sql=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return time information from osquery\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' osquery.query "select * from users;"\n    '
    return _osquery(sql)