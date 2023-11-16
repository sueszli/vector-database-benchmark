"""
Dimension Data Cloud Module
===========================

This is a cloud module for the Dimension Data Cloud,
using the existing Libcloud driver for Dimension Data.

.. code-block:: yaml

    # Note: This example is for /etc/salt/cloud.providers
    # or any file in the
    # /etc/salt/cloud.providers.d/ directory.

    my-dimensiondata-config:
      user_id: my_username
      key: myPassword!
      region: dd-na
      driver: dimensiondata

:maintainer: Anthony Shaw <anthonyshaw@apache.org>
:depends: libcloud >= 1.2.1
"""
import logging
import pprint
import socket
import salt.config as config
import salt.utils.cloud
from salt.cloud.libcloudfuncs import *
from salt.exceptions import SaltCloudExecutionFailure, SaltCloudExecutionTimeout, SaltCloudSystemExit
from salt.utils.functools import namespaced_function
from salt.utils.versions import Version
try:
    import libcloud
    from libcloud.compute.base import NodeAuthPassword, NodeDriver, NodeState
    from libcloud.compute.providers import get_driver
    from libcloud.compute.types import Provider
    from libcloud.loadbalancer.base import Member
    from libcloud.loadbalancer.providers import get_driver as get_driver_lb
    from libcloud.loadbalancer.types import Provider as Provider_lb
    if Version(libcloud.__version__) < Version('1.4.0'):
        import libcloud.security
        libcloud.security.CA_CERTS_PATH.append('/etc/ssl/certs/YaST-CA.pem')
    HAS_LIBCLOUD = True
except ImportError:
    HAS_LIBCLOUD = False
try:
    from netaddr import all_matching_cidrs
    HAS_NETADDR = True
except ImportError:
    HAS_NETADDR = False
get_size = namespaced_function(get_size, globals())
get_image = namespaced_function(get_image, globals())
avail_locations = namespaced_function(avail_locations, globals())
avail_images = namespaced_function(avail_images, globals())
avail_sizes = namespaced_function(avail_sizes, globals())
script = namespaced_function(script, globals())
destroy = namespaced_function(destroy, globals())
reboot = namespaced_function(reboot, globals())
list_nodes = namespaced_function(list_nodes, globals())
list_nodes_full = namespaced_function(list_nodes_full, globals())
list_nodes_select = namespaced_function(list_nodes_select, globals())
show_instance = namespaced_function(show_instance, globals())
get_node = namespaced_function(get_node, globals())
log = logging.getLogger(__name__)
__virtualname__ = 'dimensiondata'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set up the libcloud functions and check for dimensiondata configurations.\n    '
    if get_configured_provider() is False:
        return False
    if get_dependencies() is False:
        return False
    for (provider, details) in __opts__['providers'].items():
        if 'dimensiondata' not in details:
            continue
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
        return 10
    '\n    Return the first configured instance.\n    '
    return config.is_provider_configured(__opts__, _get_active_provider_name() or 'dimensiondata', ('user_id', 'key', 'region'))

def get_dependencies():
    if False:
        while True:
            i = 10
    "\n    Warn if dependencies aren't met.\n    "
    deps = {'libcloud': HAS_LIBCLOUD, 'netaddr': HAS_NETADDR}
    return config.check_driver_dependencies(__virtualname__, deps)

def _query_node_data(vm_, data):
    if False:
        for i in range(10):
            print('nop')
    running = False
    try:
        node = show_instance(vm_['name'], 'action')
        running = node['state'] == NodeState.RUNNING
        log.debug('Loaded node data for %s:\nname: %s\nstate: %s', vm_['name'], pprint.pformat(node['name']), node['state'])
    except Exception as err:
        log.error('Failed to get nodes list: %s', err, exc_info_on_loglevel=logging.DEBUG)
        return running
    if not running:
        return
    private = node['private_ips']
    public = node['public_ips']
    if private and (not public):
        log.warning('Private IPs returned, but not public. Checking for misidentified IPs.')
        for private_ip in private:
            private_ip = preferred_ip(vm_, [private_ip])
            if private_ip is False:
                continue
            if salt.utils.cloud.is_public_ip(private_ip):
                log.warning('%s is a public IP', private_ip)
                data.public_ips.append(private_ip)
            else:
                log.warning('%s is a private IP', private_ip)
                if private_ip not in data.private_ips:
                    data.private_ips.append(private_ip)
        if ssh_interface(vm_) == 'private_ips' and data.private_ips:
            return data
    if private:
        data.private_ips = private
        if ssh_interface(vm_) == 'private_ips':
            return data
    if public:
        data.public_ips = public
        if ssh_interface(vm_) != 'private_ips':
            return data
    log.debug('Contents of the node data:')
    log.debug(data)

def create(vm_):
    if False:
        print('Hello World!')
    '\n    Create a single VM from a data dict\n    '
    try:
        if vm_['profile'] and config.is_profile_configured(__opts__, _get_active_provider_name() or 'dimensiondata', vm_['profile']) is False:
            return False
    except AttributeError:
        pass
    __utils__['cloud.fire_event']('event', 'starting create', 'salt/cloud/{}/creating'.format(vm_['name']), args=__utils__['cloud.filter_event']('creating', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    log.info('Creating Cloud VM %s', vm_['name'])
    conn = get_conn()
    location = conn.ex_get_location_by_id(vm_['location'])
    images = conn.list_images(location=location)
    image = [x for x in images if x.id == vm_['image']][0]
    network_domains = conn.ex_list_network_domains(location=location)
    try:
        network_domain = [y for y in network_domains if y.name == vm_['network_domain']][0]
    except IndexError:
        network_domain = conn.ex_create_network_domain(location=location, name=vm_['network_domain'], plan='ADVANCED', description='')
    try:
        vlan = [y for y in conn.ex_list_vlans(location=location, network_domain=network_domain) if y.name == vm_['vlan']][0]
    except (IndexError, KeyError):
        vlan = conn.ex_list_vlans(location=location, network_domain=network_domain)[0]
    kwargs = {'name': vm_['name'], 'image': image, 'ex_description': vm_['description'], 'ex_network_domain': network_domain, 'ex_vlan': vlan, 'ex_is_started': vm_['is_started']}
    event_data = _to_event_data(kwargs)
    __utils__['cloud.fire_event']('event', 'requesting instance', 'salt/cloud/{}/requesting'.format(vm_['name']), args=__utils__['cloud.filter_event']('requesting', event_data, list(event_data)), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    initial_password = NodeAuthPassword(vm_['auth'])
    kwargs['auth'] = initial_password
    try:
        data = conn.create_node(**kwargs)
    except Exception as exc:
        log.error('Error creating %s on DIMENSIONDATA\n\nThe following exception was thrown by libcloud when trying to run the initial deployment: \n%s', vm_['name'], exc, exc_info_on_loglevel=logging.DEBUG)
        return False
    try:
        data = __utils__['cloud.wait_for_ip'](_query_node_data, update_args=(vm_, data), timeout=config.get_cloud_config_value('wait_for_ip_timeout', vm_, __opts__, default=25 * 60), interval=config.get_cloud_config_value('wait_for_ip_interval', vm_, __opts__, default=30), max_failures=config.get_cloud_config_value('wait_for_ip_max_failures', vm_, __opts__, default=60))
    except (SaltCloudExecutionTimeout, SaltCloudExecutionFailure) as exc:
        try:
            destroy(vm_['name'])
        except SaltCloudSystemExit:
            pass
        finally:
            raise SaltCloudSystemExit(str(exc))
    log.debug('VM is now running')
    if ssh_interface(vm_) == 'private_ips':
        ip_address = preferred_ip(vm_, data.private_ips)
    else:
        ip_address = preferred_ip(vm_, data.public_ips)
    log.debug('Using IP address %s', ip_address)
    if __utils__['cloud.get_salt_interface'](vm_, __opts__) == 'private_ips':
        salt_ip_address = preferred_ip(vm_, data.private_ips)
        log.info('Salt interface set to: %s', salt_ip_address)
    else:
        salt_ip_address = preferred_ip(vm_, data.public_ips)
        log.debug('Salt interface set to: %s', salt_ip_address)
    if not ip_address:
        raise SaltCloudSystemExit('No IP addresses could be found.')
    vm_['salt_host'] = salt_ip_address
    vm_['ssh_host'] = ip_address
    vm_['password'] = vm_['auth']
    ret = __utils__['cloud.bootstrap'](vm_, __opts__)
    ret.update(data.__dict__)
    if 'password' in data.extra:
        del data.extra['password']
    log.info("Created Cloud VM '%s'", vm_['name'])
    log.debug("'%s' VM creation details:\n%s", vm_['name'], pprint.pformat(data.__dict__))
    __utils__['cloud.fire_event']('event', 'created instance', 'salt/cloud/{}/created'.format(vm_['name']), args=__utils__['cloud.filter_event']('created', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    return ret

def create_lb(kwargs=None, call=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a load-balancer configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f create_lb dimensiondata \\\n            name=dev-lb port=80 protocol=http \\\n            members=w1,w2,w3 algorithm=ROUND_ROBIN\n    '
    conn = get_conn()
    if call != 'function':
        raise SaltCloudSystemExit('The create_lb function must be called with -f or --function.')
    if not kwargs or 'name' not in kwargs:
        log.error('A name must be specified when creating a health check.')
        return False
    if 'port' not in kwargs:
        log.error('A port or port-range must be specified for the load-balancer.')
        return False
    if 'networkdomain' not in kwargs:
        log.error('A network domain must be specified for the load-balancer.')
        return False
    if 'members' in kwargs:
        members = []
        ip = ''
        membersList = kwargs.get('members').split(',')
        log.debug('MemberList: %s', membersList)
        for member in membersList:
            try:
                log.debug('Member: %s', member)
                node = get_node(conn, member)
                log.debug('Node: %s', node)
                ip = node.private_ips[0]
            except Exception as err:
                log.error('Failed to get node ip: %s', err, exc_info_on_loglevel=logging.DEBUG)
            members.append(Member(ip, ip, kwargs['port']))
    else:
        members = None
    log.debug('Members: %s', members)
    networkdomain = kwargs['networkdomain']
    name = kwargs['name']
    port = kwargs['port']
    protocol = kwargs.get('protocol', None)
    algorithm = kwargs.get('algorithm', None)
    lb_conn = get_lb_conn(conn)
    network_domains = conn.ex_list_network_domains()
    network_domain = [y for y in network_domains if y.name == networkdomain][0]
    log.debug('Network Domain: %s', network_domain.id)
    lb_conn.ex_set_current_network_domain(network_domain.id)
    event_data = _to_event_data(kwargs)
    __utils__['cloud.fire_event']('event', 'create load_balancer', 'salt/cloud/loadbalancer/creating', args=event_data, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    lb = lb_conn.create_balancer(name, port, protocol, algorithm, members)
    event_data = _to_event_data(kwargs)
    __utils__['cloud.fire_event']('event', 'created load_balancer', 'salt/cloud/loadbalancer/created', args=event_data, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    return _expand_balancer(lb)

def _expand_balancer(lb):
    if False:
        i = 10
        return i + 15
    '\n    Convert the libcloud load-balancer object into something more serializable.\n    '
    ret = {}
    ret.update(lb.__dict__)
    return ret

def preferred_ip(vm_, ips):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the preferred Internet protocol. Either 'ipv4' (default) or 'ipv6'.\n    "
    proto = config.get_cloud_config_value('protocol', vm_, __opts__, default='ipv4', search_global=False)
    family = socket.AF_INET
    if proto == 'ipv6':
        family = socket.AF_INET6
    for ip in ips:
        try:
            socket.inet_pton(family, ip)
            return ip
        except Exception:
            continue
    return False

def ssh_interface(vm_):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the ssh_interface type to connect to. Either 'public_ips' (default)\n    or 'private_ips'.\n    "
    return config.get_cloud_config_value('ssh_interface', vm_, __opts__, default='public_ips', search_global=False)

def stop(name, call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Stop a VM in DimensionData.\n\n    name:\n        The name of the VM to stop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a stop vm_name\n    '
    conn = get_conn()
    node = get_node(conn, name)
    log.debug('Node of Cloud VM: %s', node)
    status = conn.ex_shutdown_graceful(node)
    log.debug('Status of Cloud VM: %s', status)
    return status

def start(name, call=None):
    if False:
        return 10
    '\n    Stop a VM in DimensionData.\n\n    :param str name:\n        The name of the VM to stop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a stop vm_name\n    '
    conn = get_conn()
    node = get_node(conn, name)
    log.debug('Node of Cloud VM: %s', node)
    status = conn.ex_start_node(node)
    log.debug('Status of Cloud VM: %s', status)
    return status

def get_conn():
    if False:
        return 10
    '\n    Return a conn object for the passed VM data\n    '
    vm_ = get_configured_provider()
    driver = get_driver(Provider.DIMENSIONDATA)
    region = config.get_cloud_config_value('region', vm_, __opts__)
    user_id = config.get_cloud_config_value('user_id', vm_, __opts__)
    key = config.get_cloud_config_value('key', vm_, __opts__)
    if key is not None:
        log.debug('DimensionData authenticating using password')
    return driver(user_id, key, region=region)

def get_lb_conn(dd_driver=None):
    if False:
        print('Hello World!')
    '\n    Return a load-balancer conn object\n    '
    vm_ = get_configured_provider()
    region = config.get_cloud_config_value('region', vm_, __opts__)
    user_id = config.get_cloud_config_value('user_id', vm_, __opts__)
    key = config.get_cloud_config_value('key', vm_, __opts__)
    if not dd_driver:
        raise SaltCloudSystemExit('Missing dimensiondata_driver for get_lb_conn method.')
    return get_driver_lb(Provider_lb.DIMENSIONDATA)(user_id, key, region=region)

def _to_event_data(obj):
    if False:
        return 10
    '\n    Convert the specified object into a form that can be serialised by msgpack as event data.\n\n    :param obj: The object to convert.\n    '
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, NodeDriver):
        return obj.name
    if isinstance(obj, list):
        return [_to_event_data(item) for item in obj]
    event_data = {}
    for attribute_name in dir(obj):
        if attribute_name.startswith('_'):
            continue
        attribute_value = getattr(obj, attribute_name)
        if callable(attribute_value):
            continue
        event_data[attribute_name] = _to_event_data(attribute_value)
    return event_data