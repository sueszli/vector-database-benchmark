"""
SoftLayer HW Cloud Module
=========================

The SoftLayer HW cloud module is used to control access to the SoftLayer
hardware cloud system

Use of this module only requires the ``apikey`` parameter. Set up the cloud
configuration at:

``/etc/salt/cloud.providers`` or ``/etc/salt/cloud.providers.d/softlayer.conf``:

.. code-block:: yaml

    my-softlayer-config:
      # SoftLayer account api key
      user: MYLOGIN
      apikey: JVkbSJDGHSDKUKSDJfhsdklfjgsjdkflhjlsdfffhgdgjkenrtuinv
      driver: softlayer_hw

The SoftLayer Python Library needs to be installed in order to use the
SoftLayer salt.cloud modules. See: https://pypi.python.org/pypi/SoftLayer

:depends: softlayer
"""
import decimal
import logging
import time
import salt.config as config
import salt.utils.cloud
from salt.exceptions import SaltCloudSystemExit
try:
    import SoftLayer
    HAS_SLLIBS = True
except ImportError:
    HAS_SLLIBS = False
log = logging.getLogger(__name__)
__virtualname__ = 'softlayer_hw'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Check for SoftLayer configurations.\n    '
    if get_configured_provider() is False:
        return False
    if get_dependencies() is False:
        return False
    return __virtualname__

def _get_active_provider_name():
    if False:
        return 10
    try:
        return __active_provider_name__.value()
    except AttributeError:
        return __active_provider_name__

def get_configured_provider():
    if False:
        i = 10
        return i + 15
    '\n    Return the first configured instance.\n    '
    return config.is_provider_configured(__opts__, _get_active_provider_name() or __virtualname__, ('apikey',))

def get_dependencies():
    if False:
        i = 10
        return i + 15
    "\n    Warn if dependencies aren't met.\n    "
    return config.check_driver_dependencies(__virtualname__, {'softlayer': HAS_SLLIBS})

def script(vm_):
    if False:
        while True:
            i = 10
    '\n    Return the script deployment object\n    '
    deploy_script = salt.utils.cloud.os_script(config.get_cloud_config_value('script', vm_, __opts__), vm_, __opts__, salt.utils.cloud.salt_config_to_yaml(salt.utils.cloud.minion_config(__opts__, vm_)))
    return deploy_script

def get_conn(service='SoftLayer_Hardware'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a conn object for the passed VM data\n    '
    client = SoftLayer.Client(username=config.get_cloud_config_value('user', get_configured_provider(), __opts__, search_global=False), api_key=config.get_cloud_config_value('apikey', get_configured_provider(), __opts__, search_global=False))
    return client[service]

def avail_locations(call=None):
    if False:
        i = 10
        return i + 15
    '\n    List all available locations\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The avail_locations function must be called with -f or --function, or with the --list-locations option')
    ret = {}
    conn = get_conn(service='SoftLayer_Product_Package')
    locations = conn.getLocations(id=50)
    for location in locations:
        ret[location['id']] = {'id': location['id'], 'name': location['name'], 'location': location['longName']}
    available = conn.getAvailableLocations(id=50)
    for location in available:
        if location.get('isAvailable', 0) == 0:
            continue
        ret[location['locationId']]['available'] = True
    return ret

def avail_sizes(call=None):
    if False:
        while True:
            i = 10
    '\n    Return a dict of all available VM sizes on the cloud provider with\n    relevant data. This data is provided in three dicts.\n\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The avail_sizes function must be called with -f or --function, or with the --list-sizes option')
    ret = {}
    conn = get_conn(service='SoftLayer_Product_Package')
    for category in conn.getCategories(id=50):
        if category['categoryCode'] != 'server_core':
            continue
        for group in category['groups']:
            for price in group['prices']:
                ret[price['id']] = price['item'].copy()
                del ret[price['id']]['id']
    return ret

def avail_images(call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a dict of all available VM images on the cloud provider.\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The avail_images function must be called with -f or --function, or with the --list-images option')
    ret = {}
    conn = get_conn(service='SoftLayer_Product_Package')
    for category in conn.getCategories(id=50):
        if category['categoryCode'] != 'os':
            continue
        for group in category['groups']:
            for price in group['prices']:
                ret[price['id']] = price['item'].copy()
                del ret[price['id']]['id']
    return ret

def get_location(vm_=None):
    if False:
        i = 10
        return i + 15
    '\n    Return the location to use, in this order:\n        - CLI parameter\n        - VM parameter\n        - Cloud profile setting\n    '
    return __opts__.get('location', config.get_cloud_config_value('location', vm_ or get_configured_provider(), __opts__, search_global=False))

def create(vm_):
    if False:
        print('Hello World!')
    '\n    Create a single VM from a data dict\n    '
    try:
        if vm_['profile'] and config.is_profile_configured(__opts__, _get_active_provider_name() or 'softlayer_hw', vm_['profile'], vm_=vm_) is False:
            return False
    except AttributeError:
        pass
    name = vm_['name']
    hostname = name
    domain = config.get_cloud_config_value('domain', vm_, __opts__, default=None)
    if domain is None:
        raise SaltCloudSystemExit('A domain name is required for the SoftLayer driver.')
    if vm_.get('use_fqdn'):
        name = '.'.join([name, domain])
        vm_['name'] = name
    __utils__['cloud.fire_event']('event', 'starting create', 'salt/cloud/{}/creating'.format(name), args=__utils__['cloud.filter_event']('creating', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    log.info('Creating Cloud VM %s', name)
    conn = get_conn(service='SoftLayer_Product_Order')
    kwargs = {'complexType': 'SoftLayer_Container_Product_Order_Hardware_Server', 'quantity': 1, 'hardware': [{'hostname': hostname, 'domain': domain}], 'packageId': 50, 'prices': [{'id': vm_['size']}, {'id': vm_['hdd']}, {'id': vm_['image']}, {'id': '905'}, {'id': '21'}, {'id': '55'}, {'id': '57'}, {'id': '58'}, {'id': '420'}, {'id': '418'}]}
    optional_products = config.get_cloud_config_value('optional_products', vm_, __opts__, default=[])
    for product in optional_products:
        kwargs['prices'].append({'id': product})
    port_speed = config.get_cloud_config_value('port_speed', vm_, __opts__, default=273)
    kwargs['prices'].append({'id': port_speed})
    bandwidth = config.get_cloud_config_value('bandwidth', vm_, __opts__, default=1800)
    kwargs['prices'].append({'id': bandwidth})
    post_uri = config.get_cloud_config_value('post_uri', vm_, __opts__, default=None)
    if post_uri:
        kwargs['prices'].append({'id': post_uri})
    vlan_id = config.get_cloud_config_value('vlan', vm_, __opts__, default=False)
    if vlan_id:
        kwargs['primaryNetworkComponent'] = {'networkVlan': {'id': vlan_id}}
    location = get_location(vm_)
    if location:
        kwargs['location'] = location
    __utils__['cloud.fire_event']('event', 'requesting instance', 'salt/cloud/{}/requesting'.format(name), args={'kwargs': __utils__['cloud.filter_event']('requesting', kwargs, list(kwargs))}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    try:
        response = conn.placeOrder(kwargs)
    except Exception as exc:
        log.error('Error creating %s on SoftLayer\n\nThe following exception was thrown when trying to run the initial deployment: \n%s', name, exc, exc_info_on_loglevel=logging.DEBUG)
        return False

    def wait_for_ip():
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait for the IP address to become available\n        '
        nodes = list_nodes_full()
        if 'primaryIpAddress' in nodes[hostname]:
            return nodes[hostname]['primaryIpAddress']
        time.sleep(1)
        return False
    ip_address = salt.utils.cloud.wait_for_fun(wait_for_ip, timeout=config.get_cloud_config_value('wait_for_fun_timeout', vm_, __opts__, default=15 * 60))
    ssh_connect_timeout = config.get_cloud_config_value('ssh_connect_timeout', vm_, __opts__, 900)
    if not salt.utils.cloud.wait_for_port(ip_address, timeout=ssh_connect_timeout):
        raise SaltCloudSystemExit('Failed to authenticate against remote ssh')
    pass_conn = get_conn(service='SoftLayer_Account')
    mask = {'virtualGuests': {'powerState': '', 'operatingSystem': {'passwords': ''}}}

    def get_passwd():
        if False:
            print('Hello World!')
        '\n        Wait for the password to become available\n        '
        node_info = pass_conn.getVirtualGuests(id=response['id'], mask=mask)
        for node in node_info:
            if node['id'] == response['id'] and 'passwords' in node['operatingSystem'] and node['operatingSystem']['passwords']:
                return node['operatingSystem']['passwords'][0]['password']
        time.sleep(5)
        return False
    passwd = salt.utils.cloud.wait_for_fun(get_passwd, timeout=config.get_cloud_config_value('wait_for_fun_timeout', vm_, __opts__, default=15 * 60))
    response['password'] = passwd
    response['public_ip'] = ip_address
    ssh_username = config.get_cloud_config_value('ssh_username', vm_, __opts__, default='root')
    vm_['ssh_host'] = ip_address
    vm_['password'] = passwd
    ret = __utils__['cloud.bootstrap'](vm_, __opts__)
    ret.update(response)
    __utils__['cloud.fire_event']('event', 'created instance', 'salt/cloud/{}/created'.format(name), args=__utils__['cloud.filter_event']('created', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    return ret

def list_nodes_full(mask='mask[id, hostname, primaryIpAddress, primaryBackendIpAddress, processorPhysicalCoreAmount, memoryCount]', call=None):
    if False:
        return 10
    '\n    Return a list of the VMs that are on the provider\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The list_nodes_full function must be called with -f or --function.')
    ret = {}
    conn = get_conn(service='SoftLayer_Account')
    response = conn.getHardware(mask=mask)
    for node in response:
        ret[node['hostname']] = node
    __utils__['cloud.cache_node_list'](ret, _get_active_provider_name().split(':')[0], __opts__)
    return ret

def list_nodes(call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of the VMs that are on the provider\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The list_nodes function must be called with -f or --function.')
    ret = {}
    nodes = list_nodes_full()
    if 'error' in nodes:
        raise SaltCloudSystemExit('An error occurred while listing nodes: {}'.format(nodes['error']['Errors']['Error']['Message']))
    for node in nodes:
        ret[node] = {'id': nodes[node]['hostname'], 'ram': nodes[node]['memoryCount'], 'cpus': nodes[node]['processorPhysicalCoreAmount']}
        if 'primaryIpAddress' in nodes[node]:
            ret[node]['public_ips'] = nodes[node]['primaryIpAddress']
        if 'primaryBackendIpAddress' in nodes[node]:
            ret[node]['private_ips'] = nodes[node]['primaryBackendIpAddress']
    return ret

def list_nodes_select(call=None):
    if False:
        print('Hello World!')
    '\n    Return a list of the VMs that are on the provider, with select fields\n    '
    return salt.utils.cloud.list_nodes_select(list_nodes_full(), __opts__['query.selection'], call)

def show_instance(name, call=None):
    if False:
        return 10
    '\n    Show the details from SoftLayer concerning a guest\n    '
    if call != 'action':
        raise SaltCloudSystemExit('The show_instance action must be called with -a or --action.')
    nodes = list_nodes_full()
    __utils__['cloud.cache_node'](nodes[name], _get_active_provider_name(), __opts__)
    return nodes[name]

def destroy(name, call=None):
    if False:
        while True:
            i = 10
    '\n    Destroy a node.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud --destroy mymachine\n    '
    if call == 'function':
        raise SaltCloudSystemExit('The destroy action must be called with -d, --destroy, -a or --action.')
    __utils__['cloud.fire_event']('event', 'destroying instance', 'salt/cloud/{}/destroying'.format(name), args={'name': name}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    node = show_instance(name, call='action')
    conn = get_conn(service='SoftLayer_Ticket')
    response = conn.createCancelServerTicket({'id': node['id'], 'reason': 'Salt Cloud Hardware Server Cancellation', 'content': 'Please cancel this server', 'cancelAssociatedItems': True, 'attachmentType': 'HARDWARE'})
    __utils__['cloud.fire_event']('event', 'destroyed instance', 'salt/cloud/{}/destroyed'.format(name), args={'name': name}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    if __opts__.get('update_cachedir', False) is True:
        __utils__['cloud.delete_minion_cachedir'](name, _get_active_provider_name().split(':')[0], __opts__)
    return response

def list_vlans(call=None):
    if False:
        i = 10
        return i + 15
    '\n    List all VLANs associated with the account\n    '
    if call != 'function':
        raise SaltCloudSystemExit('The list_vlans function must be called with -f or --function.')
    conn = get_conn(service='SoftLayer_Account')
    return conn.getNetworkVlans()

def show_pricing(kwargs=None, call=None):
    if False:
        print('Hello World!')
    '\n    Show pricing for a particular profile. This is only an estimate, based on\n    unofficial pricing sources.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-cloud -f show_pricing my-softlayerhw-config profile=my-profile\n\n    If pricing sources have not been cached, they will be downloaded. Once they\n    have been cached, they will not be updated automatically. To manually update\n    all prices, use the following command:\n\n    .. code-block:: bash\n\n        salt-cloud -f update_pricing <provider>\n\n    .. versionadded:: 2015.8.0\n    '
    profile = __opts__['profiles'].get(kwargs['profile'], {})
    if not profile:
        return {'Error': 'The requested profile was not found'}
    provider = profile.get('provider', '0:0')
    comps = provider.split(':')
    if len(comps) < 2 or comps[1] != 'softlayer_hw':
        return {'Error': 'The requested profile does not belong to Softlayer HW'}
    raw = {}
    ret = {}
    ret['per_hour'] = 0
    conn = get_conn(service='SoftLayer_Product_Item_Price')
    for item in profile:
        if item in ('profile', 'provider', 'location'):
            continue
        price = conn.getObject(id=profile[item])
        raw[item] = price
        ret['per_hour'] += decimal.Decimal(price.get('hourlyRecurringFee', 0))
    ret['per_day'] = ret['per_hour'] * 24
    ret['per_week'] = ret['per_day'] * 7
    ret['per_month'] = ret['per_day'] * 30
    ret['per_year'] = ret['per_week'] * 52
    if kwargs.get('raw', False):
        ret['_raw'] = raw
    return {profile['profile']: ret}

def show_all_prices(call=None, kwargs=None):
    if False:
        i = 10
        return i + 15
    '\n    Return a dict of all prices on the cloud provider.\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The show_all_prices function must be called with -f or --function.')
    if kwargs is None:
        kwargs = {}
    conn = get_conn(service='SoftLayer_Product_Package')
    if 'code' not in kwargs:
        return conn.getCategories(id=50)
    ret = {}
    for category in conn.getCategories(id=50):
        if category['categoryCode'] != kwargs['code']:
            continue
        for group in category['groups']:
            for price in group['prices']:
                ret[price['id']] = price['item'].copy()
                del ret[price['id']]['id']
    return ret

def show_all_categories(call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a dict of all available categories on the cloud provider.\n\n    .. versionadded:: 2016.3.0\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The show_all_categories function must be called with -f or --function.')
    conn = get_conn(service='SoftLayer_Product_Package')
    categories = []
    for category in conn.getCategories(id=50):
        categories.append(category['categoryCode'])
    return {'category_codes': categories}