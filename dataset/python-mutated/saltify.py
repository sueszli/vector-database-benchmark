"""
.. _saltify-module:

Saltify Module
==============

The Saltify module is designed to install Salt on a remote machine, virtual or
bare metal, using SSH. This module is useful for provisioning machines which
are already installed, but not Salted.

.. versionchanged:: 2018.3.0
    The wake_on_lan capability, and actions destroy, reboot, and query functions were added.

Use of this module requires some configuration in cloud profile and provider
files as described in the
:ref:`Getting Started with Saltify <getting-started-with-saltify>` documentation.
"""
import logging
import time
import salt.client
import salt.config as config
import salt.utils.cloud
from salt._compat import ipaddress
from salt.exceptions import SaltCloudException, SaltCloudSystemExit
log = logging.getLogger(__name__)
try:
    from smbprotocol.exceptions import InternalError as smbSessionError
    HAS_SMB = True
except ImportError:
    HAS_SMB = False
try:
    from requests.exceptions import ConnectionError, ConnectTimeout, InvalidSchema, ProxyError, ReadTimeout, RetryError, SSLError
    from winrm.exceptions import WinRMTransportError
    HAS_WINRM = True
except ImportError:
    HAS_WINRM = False

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Needs no special configuration\n    '
    return True

def _get_active_provider_name():
    if False:
        print('Hello World!')
    try:
        return __active_provider_name__.value()
    except AttributeError:
        return __active_provider_name__

def avail_locations(call=None):
    if False:
        print('Hello World!')
    '\n    This function returns a list of locations available.\n\n    .. code-block:: bash\n\n        salt-cloud --list-locations my-cloud-provider\n\n    [ saltify will always return an empty dictionary ]\n    '
    return {}

def avail_images(call=None):
    if False:
        print('Hello World!')
    '\n    This function returns a list of images available for this cloud provider.\n\n    .. code-block:: bash\n\n        salt-cloud --list-images saltify\n\n    returns a list of available profiles.\n\n    .. versionadded:: 2018.3.0\n\n    '
    vm_ = get_configured_provider()
    return {'Profiles': [profile for profile in vm_['profiles']]}

def avail_sizes(call=None):
    if False:
        while True:
            i = 10
    '\n    This function returns a list of sizes available for this cloud provider.\n\n    .. code-block:: bash\n\n        salt-cloud --list-sizes saltify\n\n    [ saltify always returns an empty dictionary ]\n    '
    return {}

def list_nodes(call=None):
    if False:
        i = 10
        return i + 15
    '\n    List the nodes which have salt-cloud:driver:saltify grains.\n\n    .. code-block:: bash\n\n        salt-cloud -Q\n\n    returns a list of dictionaries of defined standard fields.\n\n    .. versionadded:: 2018.3.0\n\n    '
    nodes = _list_nodes_full(call)
    return _build_required_items(nodes)

def _build_required_items(nodes):
    if False:
        i = 10
        return i + 15
    ret = {}
    for (name, grains) in nodes.items():
        if grains:
            private_ips = []
            public_ips = []
            ips = grains['ipv4'] + grains['ipv6']
            for adrs in ips:
                ip_ = ipaddress.ip_address(adrs)
                if not ip_.is_loopback:
                    if ip_.is_private:
                        private_ips.append(adrs)
                    else:
                        public_ips.append(adrs)
            ret[name] = {'id': grains['id'], 'image': grains['salt-cloud']['profile'], 'private_ips': private_ips, 'public_ips': public_ips, 'size': '', 'state': 'running'}
    return ret

def list_nodes_full(call=None):
    if False:
        return 10
    "\n    Lists complete information for all nodes.\n\n    .. code-block:: bash\n\n        salt-cloud -F\n\n    returns a list of dictionaries.\n\n    for 'saltify' minions, returns dict of grains (enhanced).\n\n    .. versionadded:: 2018.3.0\n    "
    ret = _list_nodes_full(call)
    for (key, grains) in ret.items():
        try:
            del (grains['cpu_flags'], grains['disks'], grains['pythonpath'], grains['dns'], grains['gpus'])
        except KeyError:
            pass
        except TypeError:
            del ret[key]
    reqs = _build_required_items(ret)
    for name in ret:
        ret[name].update(reqs[name])
    return ret

def _list_nodes_full(call=None):
    if False:
        print('Hello World!')
    "\n    List the nodes, ask all 'saltify' minions, return dict of grains.\n    "
    with salt.client.LocalClient() as local:
        return local.cmd('salt-cloud:driver:saltify', 'grains.items', '', tgt_type='grain')

def list_nodes_select(call=None):
    if False:
        i = 10
        return i + 15
    '\n    Return a list of the minions that have salt-cloud grains, with\n    select fields.\n    '
    return salt.utils.cloud.list_nodes_select(list_nodes_full('function'), __opts__['query.selection'], call)

def show_instance(name, call=None):
    if False:
        while True:
            i = 10
    '\n    List the a single node, return dict of grains.\n    '
    with salt.client.LocalClient() as local:
        ret = local.cmd(name, 'grains.items')
        ret.update(_build_required_items(ret))
        return ret

def create(vm_):
    if False:
        while True:
            i = 10
    '\n    if configuration parameter ``deploy`` is ``True``,\n\n        Provision a single machine, adding its keys to the salt master\n\n    else,\n\n        Test ssh connections to the machine\n\n    Configuration parameters:\n\n    - deploy:  (see above)\n    - provider:  name of entry in ``salt/cloud.providers.d/???`` file\n    - ssh_host: IP address or DNS name of the new machine\n    - ssh_username:  name used to log in to the new machine\n    - ssh_password:  password to log in (unless key_filename is used)\n    - key_filename:  (optional) SSH private key for passwordless login\n    - ssh_port: (default=22) TCP port for SSH connection\n    - wake_on_lan_mac:  (optional) hardware (MAC) address for wake on lan\n    - wol_sender_node:  (optional) salt minion to send wake on lan command\n    - wol_boot_wait:  (default=30) seconds to delay while client boots\n    - force_minion_config: (optional) replace the minion configuration files on the new machine\n\n    See also\n    :ref:`Miscellaneous Salt Cloud Options <misc-salt-cloud-options>`\n    and\n    :ref:`Getting Started with Saltify <getting-started-with-saltify>`\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -p mymachine my_new_id\n    '
    deploy_config = config.get_cloud_config_value('deploy', vm_, __opts__, default=False)
    if not config.get_cloud_config_value('ssh_host', vm_, __opts__, default=''):
        vm_['ssh_host'] = vm_['name']
    if deploy_config:
        wol_mac = config.get_cloud_config_value('wake_on_lan_mac', vm_, __opts__, default='')
        wol_host = config.get_cloud_config_value('wol_sender_node', vm_, __opts__, default='')
        if wol_mac and wol_host:
            good_ping = False
            ssh_host = config.get_cloud_config_value('ssh_host', vm_, __opts__, default='')
            with salt.client.LocalClient() as local:
                if ssh_host:
                    log.info('trying to ping %s', ssh_host)
                    count = 'n' if salt.utils.platform.is_windows() else 'c'
                    cmd = 'ping -{} 1 {}'.format(count, ssh_host)
                    good_ping = local.cmd(wol_host, 'cmd.retcode', [cmd]) == 0
                if good_ping:
                    log.info('successful ping.')
                else:
                    log.info('sending wake-on-lan to %s using node %s', wol_mac, wol_host)
                    if isinstance(wol_mac, str):
                        wol_mac = [wol_mac]
                    ret = local.cmd(wol_host, 'network.wol', wol_mac)
                    log.info('network.wol returned value %s', ret)
                    if ret and ret[wol_host]:
                        sleep_time = config.get_cloud_config_value('wol_boot_wait', vm_, __opts__, default=30)
                        if sleep_time > 0.0:
                            log.info('delaying %d seconds for boot', sleep_time)
                            time.sleep(sleep_time)
        log.info('Provisioning existing machine %s', vm_['name'])
        ret = __utils__['cloud.bootstrap'](vm_, __opts__)
    else:
        ret = _verify(vm_)
    return ret

def get_configured_provider():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the first configured instance.\n    '
    return config.is_provider_configured(__opts__, _get_active_provider_name() or 'saltify', ())

def _verify(vm_):
    if False:
        while True:
            i = 10
    '\n    Verify credentials for an existing system\n    '
    log.info('Verifying credentials for %s', vm_['name'])
    win_installer = config.get_cloud_config_value('win_installer', vm_, __opts__)
    if win_installer:
        log.debug('Testing Windows authentication method for %s', vm_['name'])
        if not HAS_SMB:
            log.error('smbprotocol library not found')
            return False
        kwargs = {'host': vm_['ssh_host'], 'username': config.get_cloud_config_value('win_username', vm_, __opts__, default='Administrator'), 'password': config.get_cloud_config_value('win_password', vm_, __opts__, default='')}
        try:
            log.debug('Testing SMB protocol for %s', vm_['name'])
            if __utils__['smb.get_conn'](**kwargs) is False:
                return False
        except smbSessionError as exc:
            log.error('Exception: %s', exc)
            return False
        use_winrm = config.get_cloud_config_value('use_winrm', vm_, __opts__, default=False)
        if use_winrm:
            log.debug('WinRM protocol requested for %s', vm_['name'])
            if not HAS_WINRM:
                log.error('WinRM library not found')
                return False
            kwargs['port'] = config.get_cloud_config_value('winrm_port', vm_, __opts__, default=5986)
            kwargs['timeout'] = 10
            try:
                log.debug('Testing WinRM protocol for %s', vm_['name'])
                return __utils__['cloud.wait_for_winrm'](**kwargs) is not None
            except (ConnectionError, ConnectTimeout, ReadTimeout, SSLError, ProxyError, RetryError, InvalidSchema, WinRMTransportError) as exc:
                log.error('Exception: %s', exc)
                return False
        return True
    else:
        log.debug('Testing SSH authentication method for %s', vm_['name'])
        kwargs = {'host': vm_['ssh_host'], 'port': config.get_cloud_config_value('ssh_port', vm_, __opts__, default=22), 'username': config.get_cloud_config_value('ssh_username', vm_, __opts__, default='root'), 'password': config.get_cloud_config_value('password', vm_, __opts__, search_global=False), 'key_filename': config.get_cloud_config_value('key_filename', vm_, __opts__, search_global=False, default=config.get_cloud_config_value('ssh_keyfile', vm_, __opts__, search_global=False, default=None)), 'gateway': vm_.get('gateway', None), 'maxtries': 1}
        log.debug('Testing SSH protocol for %s', vm_['name'])
        try:
            return __utils__['cloud.wait_for_passwd'](**kwargs) is True
        except SaltCloudException as exc:
            log.error('Exception: %s', exc)
            return False

def destroy(name, call=None):
    if False:
        while True:
            i = 10
    'Destroy a node.\n\n    .. versionadded:: 2018.3.0\n\n    Disconnect a minion from the master, and remove its keys.\n\n    Optionally, (if ``remove_config_on_destroy`` is ``True``),\n      disables salt-minion from running on the minion, and\n      erases the Salt configuration files from it.\n\n    Optionally, (if ``shutdown_on_destroy`` is ``True``),\n      orders the minion to halt.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud --destroy mymachine\n\n    '
    if call == 'function':
        raise SaltCloudSystemExit('The destroy action must be called with -d, --destroy, -a, or --action.')
    opts = __opts__
    __utils__['cloud.fire_event']('event', 'destroying instance', 'salt/cloud/{}/destroying'.format(name), args={'name': name}, sock_dir=opts['sock_dir'], transport=opts['transport'])
    vm_ = get_configured_provider()
    with salt.client.LocalClient() as local:
        my_info = local.cmd(name, 'grains.get', ['salt-cloud'])
        try:
            vm_.update(my_info[name])
        except (IndexError, TypeError):
            pass
        if config.get_cloud_config_value('remove_config_on_destroy', vm_, opts, default=True):
            ret = local.cmd(name, 'service.disable', ['salt-minion'])
            if ret and ret[name]:
                log.info('disabled salt-minion service on %s', name)
            ret = local.cmd(name, 'config.get', ['conf_file'])
            if ret and ret[name]:
                confile = ret[name]
                ret = local.cmd(name, 'file.remove', [confile])
                if ret and ret[name]:
                    log.info('removed minion %s configuration file %s', name, confile)
            ret = local.cmd(name, 'config.get', ['pki_dir'])
            if ret and ret[name]:
                pki_dir = ret[name]
                ret = local.cmd(name, 'file.remove', [pki_dir])
                if ret and ret[name]:
                    log.info('removed minion %s key files in %s', name, pki_dir)
        if config.get_cloud_config_value('shutdown_on_destroy', vm_, opts, default=False):
            ret = local.cmd(name, 'system.shutdown')
            if ret and ret[name]:
                log.info('system.shutdown for minion %s successful', name)
    __utils__['cloud.fire_event']('event', 'destroyed instance', 'salt/cloud/{}/destroyed'.format(name), args={'name': name}, sock_dir=opts['sock_dir'], transport=opts['transport'])
    return {'Destroyed': '{} was destroyed.'.format(name)}

def reboot(name, call=None):
    if False:
        return 10
    '\n    Reboot a saltify minion.\n\n    .. versionadded:: 2018.3.0\n\n    name\n        The name of the VM to reboot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a reboot vm_name\n    '
    if call != 'action':
        raise SaltCloudException('The reboot action must be called with -a or --action.')
    with salt.client.LocalClient() as local:
        return local.cmd(name, 'system.reboot')