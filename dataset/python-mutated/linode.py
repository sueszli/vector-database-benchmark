"""
The Linode Cloud Module
=======================

The Linode cloud module is used to interact with the Linode Cloud.

Provider
--------

The following provider parameters are supported:

- **apikey**: (required) The key to use to authenticate with the Linode API.
- **password**: (required) The default password to set on new VMs. Must be 8 characters with at least one lowercase, uppercase, and numeric.
- **poll_interval**: (optional) The rate of time in milliseconds to poll the Linode API for changes. Defaults to ``500``.
- **ratelimit_sleep**: (optional) The time in seconds to wait before retrying after a ratelimit has been enforced. Defaults to ``0``.

.. note::

    APIv3 usage has been removed in favor of APIv4. To move to APIv4 now,
    See the full migration guide
    here https://docs.saltproject.io/en/latest/topics/cloud/linode.html#migrating-to-apiv4.

Set up the provider configuration at ``/etc/salt/cloud.providers`` or ``/etc/salt/cloud.providers.d/linode.conf``:

.. code-block:: yaml

    my-linode-provider:
        driver: linode
        apikey: f4ZsmwtB1c7f85Jdu43RgXVDFlNjuJaeIYV8QMftTqKScEB2vSosFSr...
        password: F00barbazverylongp@ssword

Profile
-------

The following profile parameters are supported:

- **size**: (required) The size of the VM. This should be a Linode instance type ID (i.e. ``g6-standard-2``). Run ``salt-cloud -f avail_sizes my-linode-provider`` for options.
- **location**: (required) The location of the VM. This should be a Linode region (e.g. ``us-east``). Run ``salt-cloud -f avail_locations my-linode-provider`` for options.
- **image**: (required) The image to deploy the boot disk from. This should be an image ID (e.g. ``linode/ubuntu22.04``); official images start with ``linode/``. Run ``salt-cloud -f avail_images my-linode-provider`` for more options.
- **password**: (\\*required) The default password for the VM. Must be provided at the profile or provider level.
- **assign_private_ip**: (optional) Whether or not to assign a private key to the VM. Defaults to ``False``.
- **ssh_interface**: (optional) The interface with which to connect over SSH. Valid options are ``private_ips`` or ``public_ips``. Defaults to ``public_ips``.
- **ssh_pubkey**: (optional) The public key to authorize for SSH with the VM.
- **swap**: (optional) The amount of disk space to allocate for the swap partition. Defaults to ``256``.
- **clonefrom**: (optional) The name of the Linode to clone from.

Set up a profile configuration in ``/etc/salt/cloud.profiles.d/``:

.. code-block:: yaml

    my-linode-profile:
        # a minimal configuration
        provider: my-linode-provider
        size: g6-standard-1
        image: linode/ubuntu22.04
        location: us-east

    my-linode-profile-advanced:
        # an advanced configuration
        provider: my-linode-provider
        size: g6-standard-3
        image: linode/ubuntu22.04
        location: eu-west
        password: bogus123X
        assign_private_ip: true
        ssh_interface: private_ips
        ssh_pubkey: ssh-rsa AAAAB3NzaC1yc2EAAAADAQAB...
        swap_size: 512

Migrating to APIv4
------------------

You will need to generate a new token for your account. See https://www.linode.com/docs/products/tools/api/get-started/#create-an-api-token

There are a few changes to note:
- There has been a general move from label references to ID references. The profile configuration parameters ``location``, ``size``, and ``image`` have moved from being label based references to IDs. See the profile section for more information. In addition to these inputs being changed, ``avail_sizes``, ``avail_locations``, and ``avail_images`` now output options sorted by ID instead of label.
- The ``disk_size`` profile configuration parameter has been deprecated and will not be taken into account when creating new VMs while targeting APIv4.

:maintainer: Linode Developer Tools and Experience Team <dev-dx@linode.com>
:depends: requests
"""
import datetime
import json
import logging
import pprint
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
import salt.config as config
from salt._compat import ipaddress
from salt.exceptions import SaltCloudException, SaltCloudNotFound, SaltCloudSystemExit
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
log = logging.getLogger(__name__)
LASTCALL = int(time.mktime(datetime.datetime.now().timetuple()))
__virtualname__ = 'linode'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Check for Linode configs.\n    '
    if get_configured_provider() is False:
        return False
    if _get_dependencies() is False:
        return False
    return __virtualname__

def _get_active_provider_name():
    if False:
        for i in range(10):
            print('nop')
    try:
        return __active_provider_name__.value()
    except AttributeError:
        return __active_provider_name__

def get_configured_provider():
    if False:
        print('Hello World!')
    '\n    Return the first configured instance.\n    '
    return config.is_provider_configured(__opts__, _get_active_provider_name() or __virtualname__, ('apikey', 'password'))

def _get_dependencies():
    if False:
        while True:
            i = 10
    "\n    Warn if dependencies aren't met.\n    "
    deps = {'requests': HAS_REQUESTS}
    return config.check_driver_dependencies(__virtualname__, deps)

def _get_api_key():
    if False:
        i = 10
        return i + 15
    '\n    Returned the configured Linode API key.\n    '
    val = config.get_cloud_config_value('api_key', get_configured_provider(), __opts__, search_global=False, default=config.get_cloud_config_value('apikey', get_configured_provider(), __opts__, search_global=False))
    return val

def _get_ratelimit_sleep():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the configured time to wait before retrying after a ratelimit has been enforced.\n    '
    return config.get_cloud_config_value('ratelimit_sleep', get_configured_provider(), __opts__, search_global=False, default=0)

def _get_poll_interval():
    if False:
        return 10
    '\n    Return the configured interval in milliseconds to poll the Linode API for changes at.\n    '
    return config.get_cloud_config_value('poll_interval', get_configured_provider(), __opts__, search_global=False, default=500)

def _get_password(vm_):
    if False:
        return 10
    '\n    Return the password to use for a VM.\n\n    vm\\_\n        The configuration to obtain the password from.\n    '
    return config.get_cloud_config_value('password', vm_, __opts__, default=config.get_cloud_config_value('passwd', vm_, __opts__, search_global=False), search_global=False)

def _get_private_ip(vm_):
    if False:
        while True:
            i = 10
    '\n    Return True if a private ip address is requested\n    '
    return config.get_cloud_config_value('assign_private_ip', vm_, __opts__, default=False)

def _get_ssh_key_files(vm_):
    if False:
        print('Hello World!')
    '\n    Return the configured file paths of the SSH keys.\n    '
    return config.get_cloud_config_value('ssh_key_files', vm_, __opts__, search_global=False, default=[])

def _get_ssh_key(vm_):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the SSH pubkey.\n\n    vm\\_\n        The configuration to obtain the public key from.\n    '
    return config.get_cloud_config_value('ssh_pubkey', vm_, __opts__, search_global=False)

def _get_swap_size(vm_):
    if False:
        print('Hello World!')
    '\n    Returns the amount of swap space to be used in MB.\n\n    vm\\_\n        The VM profile to obtain the swap size from.\n    '
    return config.get_cloud_config_value('swap', vm_, __opts__, default=256)

def _get_ssh_keys(vm_):
    if False:
        i = 10
        return i + 15
    '\n    Return all SSH keys from ``ssh_pubkey`` and ``ssh_key_files``.\n    '
    ssh_keys = set()
    raw_pub_key = _get_ssh_key(vm_)
    if raw_pub_key is not None:
        ssh_keys.add(raw_pub_key)
    key_files = _get_ssh_key_files(vm_)
    for file in map(lambda file: Path(file).resolve(), key_files):
        if not (file.exists() or file.is_file()):
            raise SaltCloudSystemExit(f'Invalid SSH key file: {str(file)}')
        ssh_keys.add(file.read_text())
    return list(ssh_keys)

def _get_ssh_interface(vm_):
    if False:
        while True:
            i = 10
    "\n    Return the ssh_interface type to connect to. Either 'public_ips' (default)\n    or 'private_ips'.\n    "
    return config.get_cloud_config_value('ssh_interface', vm_, __opts__, default='public_ips', search_global=False)

def _validate_name(name):
    if False:
        return 10
    "\n    Checks if the provided name fits Linode's labeling parameters.\n\n    .. versionadded:: 2015.5.6\n\n    name\n        The VM name to validate\n    "
    name = str(name)
    name_length = len(name)
    regex = re.compile('^[a-zA-Z0-9][A-Za-z0-9_-]*[a-zA-Z0-9]$')
    if name_length < 3 or name_length > 48:
        ret = False
    elif not re.match(regex, name):
        ret = False
    else:
        ret = True
    if ret is False:
        log.warning('A Linode label may only contain ASCII letters or numbers, dashes, and underscores, must begin and end with letters or numbers, and be at least three characters in length.')
    return ret

class LinodeAPI(ABC):

    @abstractmethod
    def avail_images(self):
        if False:
            i = 10
            return i + 15
        'avail_images implementation'

    @abstractmethod
    def avail_locations(self):
        if False:
            i = 10
            return i + 15
        'avail_locations implementation'

    @abstractmethod
    def avail_sizes(self):
        if False:
            while True:
                i = 10
        'avail_sizes implementation'

    @abstractmethod
    def boot(self, name=None, kwargs=None):
        if False:
            i = 10
            return i + 15
        'boot implementation'

    @abstractmethod
    def clone(self, kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        'clone implementation'

    @abstractmethod
    def create_config(self, kwargs=None):
        if False:
            i = 10
            return i + 15
        'create_config implementation'

    @abstractmethod
    def create(self, vm_):
        if False:
            print('Hello World!')
        'create implementation'

    @abstractmethod
    def destroy(self, name):
        if False:
            print('Hello World!')
        'destroy implementation'

    @abstractmethod
    def get_config_id(self, kwargs=None):
        if False:
            return 10
        'get_config_id implementation'

    @abstractmethod
    def list_nodes(self):
        if False:
            while True:
                i = 10
        'list_nodes implementation'

    @abstractmethod
    def list_nodes_full(self):
        if False:
            for i in range(10):
                print('nop')
        'list_nodes_full implementation'

    @abstractmethod
    def list_nodes_min(self):
        if False:
            return 10
        'list_nodes_min implementation'

    @abstractmethod
    def reboot(self, name):
        if False:
            return 10
        'reboot implementation'

    @abstractmethod
    def show_instance(self, name):
        if False:
            print('Hello World!')
        'show_instance implementation'

    @abstractmethod
    def show_pricing(self, kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        'show_pricing implementation'

    @abstractmethod
    def start(self, name):
        if False:
            return 10
        'start implementation'

    @abstractmethod
    def stop(self, name):
        if False:
            for i in range(10):
                print('nop')
        'stop implementation'

    @abstractmethod
    def _get_linode_by_name(self, name):
        if False:
            while True:
                i = 10
        '_get_linode_by_name implementation'

    @abstractmethod
    def _get_linode_by_id(self, linode_id):
        if False:
            return 10
        '_get_linode_by_id implementation'

    def get_linode(self, kwargs=None):
        if False:
            while True:
                i = 10
        name = kwargs.get('name', None)
        linode_id = kwargs.get('linode_id', None)
        if linode_id is not None:
            return self._get_linode_by_id(linode_id)
        elif name is not None:
            return self._get_linode_by_name(name)
        raise SaltCloudSystemExit("The get_linode function requires either a 'name' or a 'linode_id'.")

    def list_nodes_select(self, call):
        if False:
            while True:
                i = 10
        return __utils__['cloud.list_nodes_select'](self.list_nodes_full(), __opts__['query.selection'], call)

class LinodeAPIv4(LinodeAPI):

    @classmethod
    def get_api_instance(cls):
        if False:
            while True:
                i = 10
        if not hasattr(cls, 'api_instance'):
            cls.api_instance = cls()
        return cls.api_instance

    def _query(self, path, method='GET', data=None, headers=None):
        if False:
            print('Hello World!')
        '\n        Make a call to the Linode API.\n        '
        api_key = _get_api_key()
        ratelimit_sleep = _get_ratelimit_sleep()
        if headers is None:
            headers = {}
        headers['Authorization'] = f'Bearer {api_key}'
        headers['Content-Type'] = 'application/json'
        headers['User-Agent'] = 'salt-cloud-linode'
        url = f'https://api.linode.com/v4{path}'
        decode = method != 'DELETE'
        result = None
        log.debug('Linode API request: %s %s', method, url)
        if data is not None:
            log.trace('Linode API request body: %s', data)
        attempt = 0
        while True:
            try:
                result = requests.request(method, url, json=data, headers=headers)
                log.debug('Linode API response status code: %d', result.status_code)
                log.trace('Linode API response body: %s', result.text)
                result.raise_for_status()
                break
            except requests.exceptions.HTTPError as exc:
                err_response = exc.response
                err_data = self._get_response_json(err_response)
                status_code = err_response.status_code
                if status_code == 429:
                    log.debug('received rate limit; retrying in %d seconds', ratelimit_sleep)
                    time.sleep(ratelimit_sleep)
                    continue
                if err_data is not None:
                    if 'error' in err_data:
                        raise SaltCloudSystemExit('Linode API reported error: {}'.format(err_data['error']))
                    elif 'errors' in err_data:
                        api_errors = err_data['errors']
                        errors = []
                        for error in err_data['errors']:
                            if 'field' in error:
                                errors.append("field '{}': {}".format(error.get('field'), error.get('reason')))
                            else:
                                errors.append(error.get('reason'))
                        raise SaltCloudSystemExit('Linode API reported error(s): {}'.format(', '.join(errors)))
                raise SaltCloudSystemExit(f'Linode API error occurred: {err_response.reason}')
        if decode:
            return self._get_response_json(result)
        return result

    def avail_images(self):
        if False:
            return 10
        response = self._query(path='/images')
        ret = {}
        for image in response['data']:
            ret[image['id']] = image
        return ret

    def avail_locations(self):
        if False:
            for i in range(10):
                print('nop')
        response = self._query(path='/regions')
        ret = {}
        for region in response['data']:
            ret[region['id']] = region
        return ret

    def avail_sizes(self):
        if False:
            i = 10
            return i + 15
        response = self._query(path='/linode/types')
        ret = {}
        for instance_type in response['data']:
            ret[instance_type['id']] = instance_type
        return ret

    def boot(self, name=None, kwargs=None):
        if False:
            return 10
        instance = self.get_linode(kwargs={'linode_id': kwargs.get('linode_id', None), 'name': name})
        config_id = kwargs.get('config_id', None)
        check_running = kwargs.get('check_running', True)
        linode_id = instance.get('id', None)
        name = instance.get('label', None)
        if check_running:
            if instance['status'] == 'running':
                raise SaltCloudSystemExit('Cannot boot Linode {0} ({1}). Linode {0} is already running.'.format(name, linode_id))
        self._query(f'/linode/instances/{linode_id}/boot', method='POST', data={'config_id': config_id})
        self._wait_for_linode_status(linode_id, 'running')
        return True

    def clone(self, kwargs=None):
        if False:
            print('Hello World!')
        linode_id = kwargs.get('linode_id', None)
        location = kwargs.get('location', None)
        size = kwargs.get('size', None)
        for item in [linode_id, location, size]:
            if item is None:
                raise SaltCloudSystemExit("The clone function requires a 'linode_id', 'location',and 'size' to be provided.")
        return self._query(f'/linode/instances/{linode_id}/clone', method='POST', data={'region': location, 'type': size})

    def create_config(self, kwargs=None):
        if False:
            i = 10
            return i + 15
        name = kwargs.get('name', None)
        linode_id = kwargs.get('linode_id', None)
        root_disk_id = kwargs.get('root_disk_id', None)
        swap_disk_id = kwargs.get('swap_disk_id', None)
        data_disk_id = kwargs.get('data_disk_id', None)
        if not name and (not linode_id):
            raise SaltCloudSystemExit("The create_config function requires either a 'name' or 'linode_id'")
        required_params = [name, linode_id, root_disk_id, swap_disk_id]
        for item in required_params:
            if item is None:
                raise SaltCloudSystemExit("The create_config functions requires a 'name', 'linode_id', 'root_disk_id', and 'swap_disk_id'.")
        devices = {'sda': {'disk_id': int(root_disk_id)}, 'sdb': {'disk_id': int(data_disk_id)} if data_disk_id is not None else None, 'sdc': {'disk_id': int(swap_disk_id)}}
        return self._query(f'/linode/instances/{linode_id}/configs', method='POST', data={'label': name, 'devices': devices})

    def create(self, vm_):
        if False:
            i = 10
            return i + 15
        name = vm_['name']
        if not _validate_name(name):
            return False
        __utils__['cloud.fire_event']('event', 'starting create', f'salt/cloud/{name}/creating', args=__utils__['cloud.filter_event']('creating', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
        log.info('Creating Cloud VM %s', name)
        result = None
        pub_ssh_keys = _get_ssh_keys(vm_)
        ssh_interface = _get_ssh_interface(vm_)
        use_private_ip = ssh_interface == 'private_ips'
        assign_private_ip = _get_private_ip(vm_) or use_private_ip
        password = _get_password(vm_)
        swap_size = _get_swap_size(vm_)
        clonefrom_name = vm_.get('clonefrom', None)
        instance_type = vm_.get('size', None)
        image = vm_.get('image', None)
        should_clone = True if clonefrom_name else False
        if should_clone:
            clone_linode = self.get_linode(kwargs={'name': clonefrom_name})
            result = clone({'linode_id': clone_linode['id'], 'location': clone_linode['region'], 'size': clone_linode['type']})
            if assign_private_ip:
                self._query('/networking/ips', method='POST', data={'type': 'ipv4', 'public': False, 'linode_id': result['id']})
        else:
            result = self._query('/linode/instances', method='POST', data={'label': name, 'type': instance_type, 'region': vm_.get('location', None), 'private_ip': assign_private_ip, 'booted': True, 'root_pass': password, 'authorized_keys': pub_ssh_keys, 'image': image, 'swap_size': swap_size})
        linode_id = result.get('id', None)
        self._wait_for_event('linode_create', 'linode', linode_id, 'finished')
        log.debug("linode '%s' has been created", name)
        if should_clone:
            self.boot(kwargs={'linode_id': linode_id})
        self._wait_for_linode_status(linode_id, 'running')
        (public_ips, private_ips) = self._get_ips(linode_id)
        data = {}
        data['id'] = linode_id
        data['name'] = result['label']
        data['size'] = result['type']
        data['state'] = result['status']
        data['ipv4'] = result['ipv4']
        data['ipv6'] = result['ipv6']
        data['public_ips'] = public_ips
        data['private_ips'] = private_ips
        if use_private_ip:
            vm_['ssh_host'] = private_ips[0]
        else:
            vm_['ssh_host'] = public_ips[0]
        __utils__['cloud.fire_event']('event', 'waiting for ssh', f'salt/cloud/{name}/waiting_for_ssh', sock_dir=__opts__['sock_dir'], args={'ip_address': vm_['ssh_host']}, transport=__opts__['transport'])
        ret = __utils__['cloud.bootstrap'](vm_, __opts__)
        ret.update(data)
        log.info("Created Cloud VM '%s'", name)
        log.debug("'%s' VM creation details:\n%s", name, pprint.pformat(data))
        __utils__['cloud.fire_event']('event', 'created instance', f'salt/cloud/{name}/created', args=__utils__['cloud.filter_event']('created', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
        return ret

    def destroy(self, name):
        if False:
            i = 10
            return i + 15
        __utils__['cloud.fire_event']('event', 'destroyed instance', f'salt/cloud/{name}/destroyed', args={'name': name}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
        if __opts__.get('update_cachedir', False) is True:
            __utils__['cloud.delete_minion_cachedir'](name, _get_active_provider_name().split(':')[0], __opts__)
        instance = self._get_linode_by_name(name)
        linode_id = instance.get('id', None)
        self._query(f'/linode/instances/{linode_id}', method='DELETE')

    def get_config_id(self, kwargs=None):
        if False:
            i = 10
            return i + 15
        name = kwargs.get('name', None)
        linode_id = kwargs.get('linode_id', None)
        if name is None and linode_id is None:
            raise SaltCloudSystemExit("The get_config_id function requires either a 'name' or a 'linode_id' to be provided.")
        if linode_id is None:
            linode_id = self.get_linode(kwargs=kwargs).get('id', None)
        response = self._query(f'/linode/instances/{linode_id}/configs')
        configs = response.get('data', [])
        return {'config_id': configs[0]['id']}

    def list_nodes_min(self):
        if False:
            print('Hello World!')
        result = self._query('/linode/instances')
        instances = result.get('data', [])
        ret = {}
        for instance in instances:
            name = instance['label']
            ret[name] = {'id': instance['id'], 'state': instance['status']}
        return ret

    def list_nodes_full(self):
        if False:
            print('Hello World!')
        return self._list_linodes(full=True)

    def list_nodes(self):
        if False:
            while True:
                i = 10
        return self._list_linodes()

    def reboot(self, name):
        if False:
            while True:
                i = 10
        instance = self._get_linode_by_name(name)
        linode_id = instance.get('id', None)
        self._query(f'/linode/instances/{linode_id}/reboot', method='POST')
        return self._wait_for_linode_status(linode_id, 'running')

    def show_instance(self, name):
        if False:
            while True:
                i = 10
        instance = self._get_linode_by_name(name)
        linode_id = instance.get('id', None)
        (public_ips, private_ips) = self._get_ips(linode_id)
        return {'id': instance['id'], 'image': instance['image'], 'name': instance['label'], 'size': instance['type'], 'state': instance['status'], 'public_ips': public_ips, 'private_ips': private_ips}

    def show_pricing(self, kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        profile = __opts__['profiles'].get(kwargs['profile'], {})
        if not profile:
            raise SaltCloudNotFound('The requested profile was not found.')
        provider = profile.get('provider', '0:0')
        comps = provider.split(':')
        if len(comps) < 2 or comps[1] != 'linode':
            raise SaltCloudException('The requested profile does not belong to Linode.')
        instance_type = self._get_linode_type(profile['size'])
        pricing = instance_type.get('price', {})
        per_hour = pricing['hourly']
        per_day = per_hour * 24
        per_week = per_day * 7
        per_month = pricing['monthly']
        per_year = per_month * 12
        return {profile['profile']: {'per_hour': per_hour, 'per_day': per_day, 'per_week': per_week, 'per_month': per_month, 'per_year': per_year}}

    def start(self, name):
        if False:
            i = 10
            return i + 15
        instance = self._get_linode_by_name(name)
        linode_id = instance.get('id', None)
        if instance['status'] == 'running':
            return {'success': True, 'action': 'start', 'state': 'Running', 'msg': 'Machine already running'}
        self._query(f'/linode/instances/{linode_id}/boot', method='POST')
        self._wait_for_linode_status(linode_id, 'running')
        return {'success': True, 'state': 'Running', 'action': 'start'}

    def stop(self, name):
        if False:
            while True:
                i = 10
        instance = self._get_linode_by_name(name)
        linode_id = instance.get('id', None)
        if instance['status'] == 'offline':
            return {'success': True, 'action': 'stop', 'state': 'Stopped', 'msg': 'Machine already stopped'}
        self._query(f'/linode/instances/{linode_id}/shutdown', method='POST')
        self._wait_for_linode_status(linode_id, 'offline')
        return {'success': True, 'state': 'Stopped', 'action': 'stop'}

    def _get_linode_by_id(self, linode_id):
        if False:
            return 10
        return self._query(f'/linode/instances/{linode_id}')

    def _get_linode_by_name(self, name):
        if False:
            print('Hello World!')
        result = self._query('/linode/instances')
        instances = result.get('data', [])
        for instance in instances:
            if instance['label'] == name:
                return instance
        raise SaltCloudNotFound(f'The specified name, {name}, could not be found.')

    def _list_linodes(self, full=False):
        if False:
            i = 10
            return i + 15
        result = self._query('/linode/instances')
        instances = result.get('data', [])
        ret = {}
        for instance in instances:
            node = {}
            node['id'] = instance['id']
            node['image'] = instance['image']
            node['name'] = instance['label']
            node['size'] = instance['type']
            node['state'] = instance['status']
            (public_ips, private_ips) = self._get_ips(node['id'])
            node['public_ips'] = public_ips
            node['private_ips'] = private_ips
            if full:
                node['extra'] = instance
            ret[instance['label']] = node
        return ret

    def _get_linode_type(self, linode_type):
        if False:
            i = 10
            return i + 15
        return self._query(f'/linode/types/{linode_type}')

    def _get_ips(self, linode_id):
        if False:
            while True:
                i = 10
        instance = self._get_linode_by_id(linode_id)
        public = []
        private = []
        for addr in instance.get('ipv4', []):
            if ipaddress.ip_address(addr).is_private:
                private.append(addr)
            else:
                public.append(addr)
        return (public, private)

    def _poll(self, description, getter, condition, timeout=None, poll_interval=None):
        if False:
            while True:
                i = 10
        '\n        Return true in handler to signal complete.\n        '
        if poll_interval is None:
            poll_interval = _get_poll_interval()
        if timeout is None:
            timeout = 120
        times = timeout * 1000 / poll_interval
        curr = 0
        while True:
            curr += 1
            result = getter()
            if condition(result):
                return True
            elif curr <= times:
                time.sleep(poll_interval / 1000)
                log.info('retrying: polling for %s...', description)
            else:
                raise SaltCloudException(f'timed out: polling for {description}')

    def _wait_for_entity_status(self, getter, status, entity_name='item', identifier='some', timeout=None):
        if False:
            print('Hello World!')
        return self._poll(f"{entity_name} (id={identifier}) status to be '{status}'", getter, lambda item: item.get('status') == status, timeout=timeout)

    def _wait_for_linode_status(self, linode_id, status, timeout=None):
        if False:
            print('Hello World!')
        return self._wait_for_entity_status(lambda : self._get_linode_by_id(linode_id), status, entity_name='linode', identifier=linode_id, timeout=timeout)

    def _check_event_status(self, event, desired_status):
        if False:
            for i in range(10):
                print('nop')
        status = event.get('status')
        action = event.get('action')
        entity = event.get('entity')
        if status == 'failed':
            raise SaltCloudSystemExit('event {} for {} (id={}) failed'.format(action, entity['type'], entity['id']))
        return status == desired_status

    def _wait_for_event(self, action, entity, entity_id, status, timeout=None):
        if False:
            return 10
        event_filter = {'+order_by': 'created', '+order': 'desc', 'seen': False, 'action': action, 'entity.id': entity_id, 'entity.type': entity}
        last_event = None
        condition = lambda event: self._check_event_status(event, status)
        while True:
            if last_event is not None:
                event_filter['+gt'] = last_event
            filter_json = json.dumps(event_filter, separators=(',', ':'))
            result = self._query('/account/events', headers={'X-Filter': filter_json})
            events = result.get('data', [])
            if len(events) == 0:
                break
            for event in events:
                event_id = event.get('id')
                event_entity = event.get('entity', None)
                last_event = event_id
                if not event_entity:
                    continue
                if not (event_entity['type'] == entity and event_entity['id'] == entity_id and (event.get('action') == action)):
                    continue
                if condition(event):
                    return True
                return self._poll(f"event {event_id} to be '{status}'", lambda : self._query(f'/account/events/{event_id}'), condition, timeout=timeout)
        return False

    def _get_response_json(self, response):
        if False:
            return 10
        json = None
        try:
            json = response.json()
        except ValueError:
            pass
        return json

def avail_images(call=None):
    if False:
        return 10
    '\n    Return available Linode images.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud --list-images my-linode-config\n        salt-cloud -f avail_images my-linode-config\n    '
    if call == 'action':
        raise SaltCloudException('The avail_images function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().avail_images()

def avail_locations(call=None):
    if False:
        while True:
            i = 10
    '\n    Return available Linode datacenter locations.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud --list-locations my-linode-config\n        salt-cloud -f avail_locations my-linode-config\n    '
    if call == 'action':
        raise SaltCloudException('The avail_locations function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().avail_locations()

def avail_sizes(call=None):
    if False:
        return 10
    '\n    Return available Linode sizes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud --list-sizes my-linode-config\n        salt-cloud -f avail_sizes my-linode-config\n    '
    if call == 'action':
        raise SaltCloudException('The avail_locations function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().avail_sizes()

def boot(name=None, kwargs=None, call=None):
    if False:
        while True:
            i = 10
    '\n    Boot a Linode.\n\n    name\n        The name of the Linode to boot. Can be used instead of ``linode_id``.\n\n    linode_id\n        The ID of the Linode to boot. If provided, will be used as an\n        alternative to ``name`` and reduces the number of API calls to\n        Linode by one. Will be preferred over ``name``.\n\n    config_id\n        The ID of the Config to boot. Required.\n\n    check_running\n        Defaults to True. If set to False, overrides the call to check if\n        the VM is running before calling the linode.boot API call. Change\n        ``check_running`` to True is useful during the boot call in the\n        create function, since the new VM will not be running yet.\n\n    Can be called as an action (which requires a name):\n\n    .. code-block:: bash\n\n        salt-cloud -a boot my-instance config_id=10\n\n    ...or as a function (which requires either a name or linode_id):\n\n    .. code-block:: bash\n\n        salt-cloud -f boot my-linode-config name=my-instance config_id=10\n        salt-cloud -f boot my-linode-config linode_id=1225876 config_id=10\n    '
    if name is None and call == 'action':
        raise SaltCloudSystemExit("The boot action requires a 'name'.")
    linode_id = kwargs.get('linode_id', None)
    config_id = kwargs.get('config_id', None)
    if call == 'function':
        name = kwargs.get('name', None)
    if name is None and linode_id is None:
        raise SaltCloudSystemExit("The boot function requires either a 'name' or a 'linode_id'.")
    return LinodeAPIv4.get_api_instance().boot(name=name, kwargs=kwargs)

def clone(kwargs=None, call=None):
    if False:
        return 10
    '\n    Clone a Linode.\n\n    linode_id\n        The ID of the Linode to clone. Required.\n\n    location\n        The location of the new Linode. Required.\n\n    size\n        The size of the new Linode (must be greater than or equal to the clone source). Required.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f clone my-linode-config linode_id=1234567 location=us-central size=g6-standard-1\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The clone function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().clone(kwargs=kwargs)

def create(vm_):
    if False:
        while True:
            i = 10
    '\n    Create a single Linode VM.\n    '
    try:
        if vm_['profile'] and config.is_profile_configured(__opts__, _get_active_provider_name() or 'linode', vm_['profile'], vm_=vm_) is False:
            return False
    except AttributeError:
        pass
    return LinodeAPIv4.get_api_instance().create(vm_)

def create_config(kwargs=None, call=None):
    if False:
        while True:
            i = 10
    '\n    Creates a Linode Configuration Profile.\n\n    name\n        The name of the VM to create the config for.\n\n    linode_id\n        The ID of the Linode to create the configuration for.\n\n    root_disk_id\n        The Root Disk ID to be used for this config.\n\n    swap_disk_id\n        The Swap Disk ID to be used for this config.\n\n    data_disk_id\n        The Data Disk ID to be used for this config.\n\n    .. versionadded:: 2016.3.0\n\n    kernel_id\n        The ID of the kernel to use for this configuration profile.\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The create_config function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().create_config(kwargs=kwargs)

def destroy(name, call=None):
    if False:
        while True:
            i = 10
    '\n    Destroys a Linode by name.\n\n    name\n        The name of VM to be be destroyed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -d vm_name\n    '
    if call == 'function':
        raise SaltCloudException('The destroy action must be called with -d, --destroy, -a or --action.')
    return LinodeAPIv4.get_api_instance().destroy(name)

def get_config_id(kwargs=None, call=None):
    if False:
        print('Hello World!')
    '\n    Returns a config_id for a given linode.\n\n    .. versionadded:: 2015.8.0\n\n    name\n        The name of the Linode for which to get the config_id. Can be used instead\n        of ``linode_id``.\n\n    linode_id\n        The ID of the Linode for which to get the config_id. Can be used instead\n        of ``name``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f get_config_id my-linode-config name=my-linode\n        salt-cloud -f get_config_id my-linode-config linode_id=1234567\n    '
    if call == 'action':
        raise SaltCloudException('The get_config_id function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().get_config_id(kwargs=kwargs)

def get_linode(kwargs=None, call=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns data for a single named Linode.\n\n    name\n        The name of the Linode for which to get data. Can be used instead\n        ``linode_id``. Note this will induce an additional API call\n        compared to using ``linode_id``.\n\n    linode_id\n        The ID of the Linode for which to get data. Can be used instead of\n        ``name``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f get_linode my-linode-config name=my-instance\n        salt-cloud -f get_linode my-linode-config linode_id=1234567\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The get_linode function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().get_linode(kwargs=kwargs)

def list_nodes(call=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of linodes, keeping only a brief listing.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -Q\n        salt-cloud --query\n        salt-cloud -f list_nodes my-linode-config\n\n    .. note::\n\n        The ``image`` label only displays information about the VM\'s distribution vendor,\n        such as "Debian" or "RHEL" and does not display the actual image name. This is\n        due to a limitation of the Linode API.\n    '
    if call == 'action':
        raise SaltCloudException('The list_nodes function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().list_nodes()

def list_nodes_full(call=None):
    if False:
        i = 10
        return i + 15
    '\n    List linodes, with all available information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -F\n        salt-cloud --full-query\n        salt-cloud -f list_nodes_full my-linode-config\n\n    .. note::\n\n        The ``image`` label only displays information about the VM\'s distribution vendor,\n        such as "Debian" or "RHEL" and does not display the actual image name. This is\n        due to a limitation of the Linode API.\n    '
    if call == 'action':
        raise SaltCloudException('The list_nodes_full function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().list_nodes_full()

def list_nodes_min(call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of the VMs that are on the provider. Only a list of VM names and\n    their state is returned. This is the minimum amount of information needed to\n    check for existing VMs.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f list_nodes_min my-linode-config\n        salt-cloud --function list_nodes_min my-linode-config\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The list_nodes_min function must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().list_nodes_min()

def list_nodes_select(call=None):
    if False:
        print('Hello World!')
    '\n    Return a list of the VMs that are on the provider, with select fields.\n    '
    return LinodeAPIv4.get_api_instance().list_nodes_select(call)

def reboot(name, call=None):
    if False:
        while True:
            i = 10
    '\n    Reboot a linode.\n\n    .. versionadded:: 2015.8.0\n\n    name\n        The name of the VM to reboot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a reboot vm_name\n    '
    if call != 'action':
        raise SaltCloudException('The show_instance action must be called with -a or --action.')
    return LinodeAPIv4.get_api_instance().reboot(name)

def show_instance(name, call=None):
    if False:
        print('Hello World!')
    '\n    Displays details about a particular Linode VM. Either a name or a linode_id must\n    be provided.\n\n    .. versionadded:: 2015.8.0\n\n    name\n        The name of the VM for which to display details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a show_instance vm_name\n\n    .. note::\n\n        The ``image`` label only displays information about the VM\'s distribution vendor,\n        such as "Debian" or "RHEL" and does not display the actual image name. This is\n        due to a limitation of the Linode API.\n    '
    if call != 'action':
        raise SaltCloudException('The show_instance action must be called with -a or --action.')
    return LinodeAPIv4.get_api_instance().show_instance(name)

def show_pricing(kwargs=None, call=None):
    if False:
        while True:
            i = 10
    '\n    Show pricing for a particular profile. This is only an estimate, based on\n    unofficial pricing sources.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f show_pricing my-linode-config profile=my-linode-profile\n    '
    if call != 'function':
        raise SaltCloudException('The show_instance action must be called with -f or --function.')
    return LinodeAPIv4.get_api_instance().show_pricing(kwargs=kwargs)

def start(name, call=None):
    if False:
        while True:
            i = 10
    '\n    Start a VM in Linode.\n\n    name\n        The name of the VM to start.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a stop vm_name\n    '
    if call != 'action':
        raise SaltCloudException('The start action must be called with -a or --action.')
    return LinodeAPIv4.get_api_instance().start(name)

def stop(name, call=None):
    if False:
        while True:
            i = 10
    '\n    Stop a VM in Linode.\n\n    name\n        The name of the VM to stop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a stop vm_name\n    '
    if call != 'action':
        raise SaltCloudException('The stop action must be called with -a or --action.')
    return LinodeAPIv4.get_api_instance().stop(name)