"""
Pyrax Cloud Module
==================

PLEASE NOTE: This module is currently in early development, and considered to
be experimental and unstable. It is not recommended for production use. Unless
you are actively developing code in this module, you should use the OpenStack
module instead.
"""
import salt.config as config
import salt.utils.data
from salt.utils.openstack import pyrax as suop
__virtualname__ = 'pyrax'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Check for Pyrax configurations\n    '
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
    return config.is_provider_configured(__opts__, _get_active_provider_name() or __virtualname__, ('username', 'identity_url', 'compute_region'))

def get_dependencies():
    if False:
        for i in range(10):
            print('nop')
    "\n    Warn if dependencies aren't met.\n    "
    return config.check_driver_dependencies(__virtualname__, {'pyrax': suop.HAS_PYRAX})

def get_conn(conn_type):
    if False:
        return 10
    '\n    Return a conn object for the passed VM data\n    '
    vm_ = get_configured_provider()
    kwargs = vm_.copy()
    kwargs['username'] = vm_['username']
    kwargs['auth_endpoint'] = vm_.get('identity_url', None)
    kwargs['region'] = vm_['compute_region']
    conn = getattr(suop, conn_type)
    return conn(**kwargs)

def queues_exists(call, kwargs):
    if False:
        for i in range(10):
            print('nop')
    conn = get_conn('RackspaceQueues')
    return conn.exists(kwargs['name'])

def queues_show(call, kwargs):
    if False:
        for i in range(10):
            print('nop')
    conn = get_conn('RackspaceQueues')
    return salt.utils.data.simple_types_filter(conn.show(kwargs['name']).__dict__)

def queues_create(call, kwargs):
    if False:
        print('Hello World!')
    conn = get_conn('RackspaceQueues')
    if conn.create(kwargs['name']):
        return salt.utils.data.simple_types_filter(conn.show(kwargs['name']).__dict__)
    else:
        return {}

def queues_delete(call, kwargs):
    if False:
        i = 10
        return i + 15
    conn = get_conn('RackspaceQueues')
    if conn.delete(kwargs['name']):
        return {}
    else:
        return salt.utils.data.simple_types_filter(conn.show(kwargs['name'].__dict__))