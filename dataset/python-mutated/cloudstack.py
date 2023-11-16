"""
CloudStack Cloud Module
=======================

The CloudStack cloud module is used to control access to a CloudStack based
Public Cloud.

:depends: libcloud >= 0.15

Use of this module requires the ``apikey``, ``secretkey``, ``host`` and
``path`` parameters.

.. code-block:: yaml

    my-cloudstack-cloud-config:
      apikey: <your api key >
      secretkey: <your secret key >
      host: localhost
      path: /client/api
      driver: cloudstack

"""
import logging
import pprint
import salt.config as config
import salt.utils.cloud
import salt.utils.event
from salt.cloud.libcloudfuncs import *
from salt.exceptions import SaltCloudSystemExit
from salt.utils.functools import namespaced_function
from salt.utils.versions import Version
try:
    from libcloud.compute.drivers.cloudstack import CloudStackNetwork
    if Version(libcloud.__version__) < Version('1.4.0'):
        import libcloud.security
        libcloud.security.CA_CERTS_PATH.append('/etc/ssl/certs/YaST-CA.pem')
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
log = logging.getLogger(__name__)
get_node = namespaced_function(get_node, globals())
get_size = namespaced_function(get_size, globals())
get_image = namespaced_function(get_image, globals())
avail_locations = namespaced_function(avail_locations, globals())
avail_images = namespaced_function(avail_images, globals())
avail_sizes = namespaced_function(avail_sizes, globals())
script = namespaced_function(script, globals())
list_nodes = namespaced_function(list_nodes, globals())
list_nodes_full = namespaced_function(list_nodes_full, globals())
list_nodes_select = namespaced_function(list_nodes_select, globals())
show_instance = namespaced_function(show_instance, globals())
__virtualname__ = 'cloudstack'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set up the libcloud functions and check for CloudStack configurations.\n    '
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
        print('Hello World!')
    '\n    Return the first configured instance.\n    '
    return config.is_provider_configured(__opts__, _get_active_provider_name() or __virtualname__, ('apikey', 'secretkey', 'host', 'path'))

def get_dependencies():
    if False:
        for i in range(10):
            print('nop')
    "\n    Warn if dependencies aren't met.\n    "
    return config.check_driver_dependencies(__virtualname__, {'libcloud': HAS_LIBS})

def get_conn():
    if False:
        print('Hello World!')
    '\n    Return a conn object for the passed VM data\n    '
    driver = get_driver(Provider.CLOUDSTACK)
    verify_ssl_cert = config.get_cloud_config_value('verify_ssl_cert', get_configured_provider(), __opts__, default=True, search_global=False)
    if verify_ssl_cert is False:
        try:
            import libcloud.security
            libcloud.security.VERIFY_SSL_CERT = False
        except (ImportError, AttributeError):
            raise SaltCloudSystemExit('Could not disable SSL certificate verification. Not loading module.')
    return driver(key=config.get_cloud_config_value('apikey', get_configured_provider(), __opts__, search_global=False), secret=config.get_cloud_config_value('secretkey', get_configured_provider(), __opts__, search_global=False), secure=config.get_cloud_config_value('secure', get_configured_provider(), __opts__, default=True, search_global=False), host=config.get_cloud_config_value('host', get_configured_provider(), __opts__, search_global=False), path=config.get_cloud_config_value('path', get_configured_provider(), __opts__, search_global=False), port=config.get_cloud_config_value('port', get_configured_provider(), __opts__, default=None, search_global=False))

def get_location(conn, vm_):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the node location to use\n    '
    locations = conn.list_locations()
    loc = config.get_cloud_config_value('location', vm_, __opts__, default=2)
    for location in locations:
        if str(loc) in (str(location.id), str(location.name)):
            return location

def get_security_groups(conn, vm_):
    if False:
        return 10
    "\n    Return a list of security groups to use, defaulting to ['default']\n    "
    securitygroup_enabled = config.get_cloud_config_value('securitygroup_enabled', vm_, __opts__, default=True)
    if securitygroup_enabled:
        return config.get_cloud_config_value('securitygroup', vm_, __opts__, default=['default'])
    else:
        return False

def get_password(vm_):
    if False:
        while True:
            i = 10
    '\n    Return the password to use\n    '
    return config.get_cloud_config_value('password', vm_, __opts__, default=config.get_cloud_config_value('passwd', vm_, __opts__, search_global=False), search_global=False)

def get_key():
    if False:
        return 10
    '\n    Returns the ssh private key for VM access\n    '
    return config.get_cloud_config_value('private_key', get_configured_provider(), __opts__, search_global=False)

def get_keypair(vm_):
    if False:
        while True:
            i = 10
    '\n    Return the keypair to use\n    '
    keypair = config.get_cloud_config_value('keypair', vm_, __opts__)
    if keypair:
        return keypair
    else:
        return False

def get_ip(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the IP address of the VM\n    If the VM has  public IP as defined by libcloud module then use it\n    Otherwise try to extract the private IP and use that one.\n    '
    try:
        ip = data.public_ips[0]
    except Exception:
        ip = data.private_ips[0]
    return ip

def get_networkid(vm_):
    if False:
        while True:
            i = 10
    '\n    Return the networkid to use, only valid for Advanced Zone\n    '
    networkid = config.get_cloud_config_value('networkid', vm_, __opts__)
    if networkid is not None:
        return networkid
    else:
        return False

def get_project(conn, vm_):
    if False:
        i = 10
        return i + 15
    '\n    Return the project to use.\n    '
    try:
        projects = conn.ex_list_projects()
    except AttributeError:
        log.warning('Cannot get projects, you may need to update libcloud to 0.15 or later')
        return False
    projid = config.get_cloud_config_value('projectid', vm_, __opts__)
    if not projid:
        return False
    for project in projects:
        if str(projid) in (str(project.id), str(project.name)):
            return project
    log.warning("Couldn't find project %s in projects", projid)
    return False

def create(vm_):
    if False:
        print('Hello World!')
    '\n    Create a single VM from a data dict\n    '
    try:
        if vm_['profile'] and config.is_profile_configured(__opts__, _get_active_provider_name() or 'cloudstack', vm_['profile'], vm_=vm_) is False:
            return False
    except AttributeError:
        pass
    __utils__['cloud.fire_event']('event', 'starting create', 'salt/cloud/{}/creating'.format(vm_['name']), sock_dir=__opts__['sock_dir'], args=__utils__['cloud.filter_event']('creating', vm_, ['name', 'profile', 'provider', 'driver']), transport=__opts__['transport'])
    log.info('Creating Cloud VM %s', vm_['name'])
    conn = get_conn()
    kwargs = {'name': vm_['name'], 'image': get_image(conn, vm_), 'size': get_size(conn, vm_), 'location': get_location(conn, vm_)}
    sg = get_security_groups(conn, vm_)
    if sg is not False:
        kwargs['ex_security_groups'] = sg
    if get_keypair(vm_) is not False:
        kwargs['ex_keyname'] = get_keypair(vm_)
    if get_networkid(vm_) is not False:
        kwargs['networkids'] = get_networkid(vm_)
        kwargs['networks'] = (CloudStackNetwork(None, None, None, kwargs['networkids'], None, None),)
    if get_project(conn, vm_) is not False:
        kwargs['project'] = get_project(conn, vm_)
    event_data = kwargs.copy()
    event_data['image'] = kwargs['image'].name
    event_data['size'] = kwargs['size'].name
    __utils__['cloud.fire_event']('event', 'requesting instance', 'salt/cloud/{}/requesting'.format(vm_['name']), sock_dir=__opts__['sock_dir'], args={'kwargs': __utils__['cloud.filter_event']('requesting', event_data, ['name', 'profile', 'provider', 'driver', 'image', 'size'])}, transport=__opts__['transport'])
    displayname = cloudstack_displayname(vm_)
    if displayname:
        kwargs['ex_displayname'] = displayname
    else:
        kwargs['ex_displayname'] = kwargs['name']
    volumes = {}
    ex_blockdevicemappings = block_device_mappings(vm_)
    if ex_blockdevicemappings:
        for ex_blockdevicemapping in ex_blockdevicemappings:
            if 'VirtualName' not in ex_blockdevicemapping:
                ex_blockdevicemapping['VirtualName'] = '{}-{}'.format(vm_['name'], len(volumes))
            __utils__['cloud.fire_event']('event', 'requesting volume', 'salt/cloud/{}/requesting'.format(ex_blockdevicemapping['VirtualName']), sock_dir=__opts__['sock_dir'], args={'kwargs': {'name': ex_blockdevicemapping['VirtualName'], 'device': ex_blockdevicemapping['DeviceName'], 'size': ex_blockdevicemapping['VolumeSize']}})
            try:
                volumes[ex_blockdevicemapping['DeviceName']] = conn.create_volume(ex_blockdevicemapping['VolumeSize'], ex_blockdevicemapping['VirtualName'])
            except Exception as exc:
                log.error('Error creating volume %s on CLOUDSTACK\n\nThe following exception was thrown by libcloud when trying to requesting a volume: \n%s', ex_blockdevicemapping['VirtualName'], exc, exc_info_on_loglevel=logging.DEBUG)
                return False
    else:
        ex_blockdevicemapping = {}
    try:
        data = conn.create_node(**kwargs)
    except Exception as exc:
        log.error('Error creating %s on CLOUDSTACK\n\nThe following exception was thrown by libcloud when trying to run the initial deployment: \n%s', vm_['name'], exc, exc_info_on_loglevel=logging.DEBUG)
        return False
    for device_name in volumes:
        try:
            conn.attach_volume(data, volumes[device_name], device_name)
        except Exception as exc:
            log.error('Error attaching volume %s on CLOUDSTACK\n\nThe following exception was thrown by libcloud when trying to attach a volume: \n%s', ex_blockdevicemapping.get('VirtualName', 'UNKNOWN'), exc, exc_info=log.isEnabledFor(logging.DEBUG))
            return False
    ssh_username = config.get_cloud_config_value('ssh_username', vm_, __opts__, default='root')
    vm_['ssh_host'] = get_ip(data)
    vm_['password'] = data.extra['password']
    vm_['key_filename'] = get_key()
    ret = __utils__['cloud.bootstrap'](vm_, __opts__)
    ret.update(data.__dict__)
    if 'password' in data.extra:
        del data.extra['password']
    log.info("Created Cloud VM '%s'", vm_['name'])
    log.debug("'%s' VM creation details:\n%s", vm_['name'], pprint.pformat(data.__dict__))
    __utils__['cloud.fire_event']('event', 'created instance', 'salt/cloud/{}/created'.format(vm_['name']), sock_dir=__opts__['sock_dir'], args=__utils__['cloud.filter_event']('created', vm_, ['name', 'profile', 'provider', 'driver']), transport=__opts__['transport'])
    return ret

def destroy(name, conn=None, call=None):
    if False:
        while True:
            i = 10
    '\n    Delete a single VM, and all of its volumes\n    '
    if call == 'function':
        raise SaltCloudSystemExit('The destroy action must be called with -d, --destroy, -a or --action.')
    __utils__['cloud.fire_event']('event', 'destroying instance', 'salt/cloud/{}/destroying'.format(name), sock_dir=__opts__['sock_dir'], args={'name': name})
    if not conn:
        conn = get_conn()
    node = get_node(conn, name)
    if node is None:
        log.error('Unable to find the VM %s', name)
    volumes = conn.list_volumes(node)
    if volumes is None:
        log.error('Unable to find volumes of the VM %s', name)
    for volume in volumes:
        if volume.extra['volume_type'] != 'DATADISK':
            log.info('Ignoring volume type %s: %s', volume.extra['volume_type'], volume.name)
            continue
        log.info('Detaching volume: %s', volume.name)
        __utils__['cloud.fire_event']('event', 'detaching volume', 'salt/cloud/{}/detaching'.format(volume.name), sock_dir=__opts__['sock_dir'], args={'name': volume.name})
        if not conn.detach_volume(volume):
            log.error('Failed to Detach volume: %s', volume.name)
            return False
        log.info('Detached volume: %s', volume.name)
        __utils__['cloud.fire_event']('event', 'detached volume', 'salt/cloud/{}/detached'.format(volume.name), sock_dir=__opts__['sock_dir'], args={'name': volume.name})
        log.info('Destroying volume: %s', volume.name)
        __utils__['cloud.fire_event']('event', 'destroying volume', 'salt/cloud/{}/destroying'.format(volume.name), sock_dir=__opts__['sock_dir'], args={'name': volume.name})
        if not conn.destroy_volume(volume):
            log.error('Failed to Destroy volume: %s', volume.name)
            return False
        log.info('Destroyed volume: %s', volume.name)
        __utils__['cloud.fire_event']('event', 'destroyed volume', 'salt/cloud/{}/destroyed'.format(volume.name), sock_dir=__opts__['sock_dir'], args={'name': volume.name})
    log.info('Destroying VM: %s', name)
    ret = conn.destroy_node(node)
    if not ret:
        log.error('Failed to Destroy VM: %s', name)
        return False
    log.info('Destroyed VM: %s', name)
    event = salt.utils.event.SaltEvent('master', __opts__['sock_dir'])
    __utils__['cloud.fire_event']('event', 'destroyed instance', 'salt/cloud/{}/destroyed'.format(name), sock_dir=__opts__['sock_dir'], args={'name': name})
    if __opts__['delete_sshkeys'] is True:
        salt.utils.cloud.remove_sshkey(node.public_ips[0])
    return True

def block_device_mappings(vm_):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the block device mapping:\n\n    ::\n\n        [{'DeviceName': '/dev/sdb', 'VirtualName': 'ephemeral0'},\n          {'DeviceName': '/dev/sdc', 'VirtualName': 'ephemeral1'}]\n    "
    return config.get_cloud_config_value('block_device_mappings', vm_, __opts__, search_global=True)

def cloudstack_displayname(vm_):
    if False:
        i = 10
        return i + 15
    '\n    Return display name of VM:\n\n    ::\n        "minion1"\n    '
    return config.get_cloud_config_value('cloudstack_displayname', vm_, __opts__, search_global=True)