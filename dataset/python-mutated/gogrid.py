"""
GoGrid Cloud Module
====================

The GoGrid cloud module. This module interfaces with the gogrid public cloud
service. To use Salt Cloud with GoGrid log into the GoGrid web interface and
create an api key. Do this by clicking on "My Account" and then going to the
API Keys tab.

Set up the cloud configuration at ``/etc/salt/cloud.providers`` or
``/etc/salt/cloud.providers.d/gogrid.conf``:

.. code-block:: yaml

    my-gogrid-config:
      # The generated api key to use
      apikey: asdff7896asdh789
      # The apikey's shared secret
      sharedsecret: saltybacon
      driver: gogrid

.. note::

    A Note about using Map files with GoGrid:

    Due to limitations in the GoGrid API, instances cannot be provisioned in parallel
    with the GoGrid driver. Map files will work with GoGrid, but the ``-P``
    argument should not be used on maps referencing GoGrid instances.

.. note::

    A Note about using Map files with GoGrid:

    Due to limitations in the GoGrid API, instances cannot be provisioned in parallel
    with the GoGrid driver. Map files will work with GoGrid, but the ``-P``
    argument should not be used on maps referencing GoGrid instances.

"""
import logging
import pprint
import time
import salt.config as config
import salt.utils.cloud
import salt.utils.hashutils
from salt.exceptions import SaltCloudException, SaltCloudSystemExit
log = logging.getLogger(__name__)
__virtualname__ = 'gogrid'

def __virtual__():
    if False:
        return 10
    '\n    Check for GoGrid configs\n    '
    if get_configured_provider() is False:
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
        return 10
    '\n    Return the first configured instance.\n    '
    return config.is_provider_configured(__opts__, _get_active_provider_name() or __virtualname__, ('apikey', 'sharedsecret'))

def create(vm_):
    if False:
        print('Hello World!')
    '\n    Create a single VM from a data dict\n    '
    try:
        if vm_['profile'] and config.is_profile_configured(__opts__, _get_active_provider_name() or 'gogrid', vm_['profile'], vm_=vm_) is False:
            return False
    except AttributeError:
        pass
    __utils__['cloud.fire_event']('event', 'starting create', 'salt/cloud/{}/creating'.format(vm_['name']), args=__utils__['cloud.filter_event']('creating', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    if len(vm_['name']) > 20:
        raise SaltCloudException('VM names must not be longer than 20 characters')
    log.info('Creating Cloud VM %s', vm_['name'])
    image_id = avail_images()[vm_['image']]['id']
    if 'assign_public_ip' in vm_:
        host_ip = vm_['assign_public_ip']
    else:
        public_ips = list_public_ips()
        if not public_ips:
            raise SaltCloudException('No more IPs available')
        host_ip = next(iter(public_ips))
    create_kwargs = {'name': vm_['name'], 'image': image_id, 'ram': vm_['size'], 'ip': host_ip}
    __utils__['cloud.fire_event']('event', 'requesting instance', 'salt/cloud/{}/requesting'.format(vm_['name']), args={'kwargs': __utils__['cloud.filter_event']('requesting', create_kwargs, list(create_kwargs))}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    try:
        data = _query('grid', 'server/add', args=create_kwargs)
    except Exception:
        log.error('Error creating %s on GOGRID\n\nThe following exception was thrown when trying to run the initial deployment:\n', vm_['name'], exc_info_on_loglevel=logging.DEBUG)
        return False
    ssh_username = config.get_cloud_config_value('ssh_username', vm_, __opts__, default='root')

    def wait_for_apipass():
        if False:
            print('Hello World!')
        '\n        Wait for the password to become available, via the API\n        '
        try:
            passwords = list_passwords()
            return passwords[vm_['name']][0]['password']
        except KeyError:
            pass
        time.sleep(5)
        return False
    vm_['password'] = salt.utils.cloud.wait_for_fun(wait_for_apipass, timeout=config.get_cloud_config_value('wait_for_fun_timeout', vm_, __opts__, default=15 * 60))
    vm_['ssh_host'] = host_ip
    ret = __utils__['cloud.bootstrap'](vm_, __opts__)
    ret.update(data)
    log.info("Created Cloud VM '%s'", vm_['name'])
    log.debug("'%s' VM creation details:\n%s", vm_['name'], pprint.pformat(data))
    __utils__['cloud.fire_event']('event', 'created instance', 'salt/cloud/{}/created'.format(vm_['name']), args=__utils__['cloud.filter_event']('created', vm_, ['name', 'profile', 'provider', 'driver']), sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    return ret

def list_nodes(full=False, call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List of nodes, keeping only a brief listing\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -Q\n    '
    if call == 'action':
        raise SaltCloudSystemExit('The list_nodes function must be called with -f or --function.')
    ret = {}
    nodes = list_nodes_full('function')
    if full:
        return nodes
    for node in nodes:
        ret[node] = {}
        for item in ('id', 'image', 'size', 'public_ips', 'private_ips', 'state'):
            ret[node][item] = nodes[node][item]
    return ret

def list_nodes_full(call=None):
    if False:
        return 10
    '\n    List nodes, with all available information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -F\n    '
    response = _query('grid', 'server/list')
    ret = {}
    for item in response['list']:
        name = item['name']
        ret[name] = item
        ret[name]['image_info'] = item['image']
        ret[name]['image'] = item['image']['friendlyName']
        ret[name]['size'] = item['ram']['name']
        ret[name]['public_ips'] = [item['ip']['ip']]
        ret[name]['private_ips'] = []
        ret[name]['state_info'] = item['state']
        if 'active' in item['state']['description']:
            ret[name]['state'] = 'RUNNING'
    return ret

def list_nodes_select(call=None):
    if False:
        while True:
            i = 10
    '\n    Return a list of the VMs that are on the provider, with select fields\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -S\n    '
    return salt.utils.cloud.list_nodes_select(list_nodes_full('function'), __opts__['query.selection'], call)

def avail_locations():
    if False:
        print('Hello World!')
    '\n    Available locations\n    '
    response = list_common_lookups(kwargs={'lookup': 'ip.datacenter'})
    ret = {}
    for item in response['list']:
        name = item['name']
        ret[name] = item
    return ret

def avail_sizes():
    if False:
        while True:
            i = 10
    '\n    Available sizes\n    '
    response = list_common_lookups(kwargs={'lookup': 'server.ram'})
    ret = {}
    for item in response['list']:
        name = item['name']
        ret[name] = item
    return ret

def avail_images():
    if False:
        print('Hello World!')
    '\n    Available images\n    '
    response = _query('grid', 'image/list')
    ret = {}
    for item in response['list']:
        name = item['friendlyName']
        ret[name] = item
    return ret

def list_passwords(kwargs=None, call=None):
    if False:
        return 10
    '\n    List all password on the account\n\n    .. versionadded:: 2015.8.0\n    '
    response = _query('support', 'password/list')
    ret = {}
    for item in response['list']:
        if 'server' in item:
            server = item['server']['name']
            if server not in ret:
                ret[server] = []
            ret[server].append(item)
    return ret

def list_public_ips(kwargs=None, call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List all available public IPs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f list_public_ips <provider>\n\n    To list unavailable (assigned) IPs, use:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -f list_public_ips <provider> state=assigned\n\n    .. versionadded:: 2015.8.0\n    '
    if kwargs is None:
        kwargs = {}
    args = {}
    if 'state' in kwargs:
        if kwargs['state'] == 'assigned':
            args['ip.state'] = 'Assigned'
        else:
            args['ip.state'] = 'Unassigned'
    else:
        args['ip.state'] = 'Unassigned'
    args['ip.type'] = 'Public'
    response = _query('grid', 'ip/list', args=args)
    ret = {}
    for item in response['list']:
        name = item['ip']
        ret[name] = item
    return ret

def list_common_lookups(kwargs=None, call=None):
    if False:
        i = 10
        return i + 15
    '\n    List common lookups for a particular type of item\n\n    .. versionadded:: 2015.8.0\n    '
    if kwargs is None:
        kwargs = {}
    args = {}
    if 'lookup' in kwargs:
        args['lookup'] = kwargs['lookup']
    response = _query('common', 'lookup/list', args=args)
    return response

def destroy(name, call=None):
    if False:
        print('Hello World!')
    '\n    Destroy a machine by name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -d vm_name\n    '
    if call == 'function':
        raise SaltCloudSystemExit('The destroy action must be called with -d, --destroy, -a or --action.')
    __utils__['cloud.fire_event']('event', 'destroying instance', 'salt/cloud/{}/destroying'.format(name), args={'name': name}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    response = _query('grid', 'server/delete', args={'name': name})
    __utils__['cloud.fire_event']('event', 'destroyed instance', 'salt/cloud/{}/destroyed'.format(name), args={'name': name}, sock_dir=__opts__['sock_dir'], transport=__opts__['transport'])
    if __opts__.get('update_cachedir', False) is True:
        __utils__['cloud.delete_minion_cachedir'](name, _get_active_provider_name().split(':')[0], __opts__)
    return response

def reboot(name, call=None):
    if False:
        i = 10
        return i + 15
    '\n    Reboot a machine by name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a reboot vm_name\n\n    .. versionadded:: 2015.8.0\n    '
    return _query('grid', 'server/power', args={'name': name, 'power': 'restart'})

def stop(name, call=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Stop a machine by name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a stop vm_name\n\n    .. versionadded:: 2015.8.0\n    '
    return _query('grid', 'server/power', args={'name': name, 'power': 'stop'})

def start(name, call=None):
    if False:
        while True:
            i = 10
    '\n    Start a machine by name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a start vm_name\n\n    .. versionadded:: 2015.8.0\n    '
    return _query('grid', 'server/power', args={'name': name, 'power': 'start'})

def show_instance(name, call=None):
    if False:
        print('Hello World!')
    '\n    Start a machine by name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-cloud -a show_instance vm_name\n\n    .. versionadded:: 2015.8.0\n    '
    response = _query('grid', 'server/get', args={'name': name})
    ret = {}
    for item in response['list']:
        name = item['name']
        ret[name] = item
        ret[name]['image_info'] = item['image']
        ret[name]['image'] = item['image']['friendlyName']
        ret[name]['size'] = item['ram']['name']
        ret[name]['public_ips'] = [item['ip']['ip']]
        ret[name]['private_ips'] = []
        ret[name]['state_info'] = item['state']
        if 'active' in item['state']['description']:
            ret[name]['state'] = 'RUNNING'
    return ret

def _query(action=None, command=None, args=None, method='GET', header_dict=None, data=None):
    if False:
        return 10
    '\n    Make a web call to GoGrid\n\n    .. versionadded:: 2015.8.0\n    '
    vm_ = get_configured_provider()
    apikey = config.get_cloud_config_value('apikey', vm_, __opts__, search_global=False)
    sharedsecret = config.get_cloud_config_value('sharedsecret', vm_, __opts__, search_global=False)
    path = 'https://api.gogrid.com/api/'
    if action:
        path += action
    if command:
        path += '/{}'.format(command)
    log.debug('GoGrid URL: %s', path)
    if not isinstance(args, dict):
        args = {}
    epoch = str(int(time.time()))
    hashtext = ''.join((apikey, sharedsecret, epoch))
    args['sig'] = salt.utils.hashutils.md5_digest(hashtext)
    args['format'] = 'json'
    args['v'] = '1.0'
    args['api_key'] = apikey
    if header_dict is None:
        header_dict = {}
    if method != 'POST':
        header_dict['Accept'] = 'application/json'
    decode = True
    if method == 'DELETE':
        decode = False
    return_content = None
    result = salt.utils.http.query(path, method, params=args, data=data, header_dict=header_dict, decode=decode, decode_type='json', text=True, status=True, opts=__opts__)
    log.debug('GoGrid Response Status Code: %s', result['status'])
    return result['dict']