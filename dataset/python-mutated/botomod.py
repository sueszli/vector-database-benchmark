"""
Boto Common Utils
=================

Note: This module depends on the dicts packed by the loader and,
therefore, must be accessed via the loader or from the __utils__ dict.

This module provides common functionality for the boto execution modules.
The expected usage is to call `assign_funcs` from the `__virtual__` function
of the module. This will bring properly initialized partials of  `_get_conn`
and `_cache_id` into the module's namespace.

Example Usage:

    .. code-block:: python

        def __virtual__():
            __utils__['boto.assign_funcs'](__name__, 'vpc')

        def test():
            conn = _get_conn()
            vpc_id = _cache_id('test-vpc')

.. versionadded:: 2015.8.0
"""
import hashlib
import logging
import sys
from functools import partial
import salt.loader.context
import salt.utils.stringutils
import salt.utils.versions
from salt.exceptions import SaltInvocationError
from salt.loader import minion_mods
try:
    import boto
    import boto.exception
    logging.getLogger('boto').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
log = logging.getLogger(__name__)
__salt__ = None
__virtualname__ = 'boto'
__salt_loader__ = salt.loader.context.LoaderContext()
__context__ = __salt_loader__.named_context('__context__', {})

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    has_boto_requirements = salt.utils.versions.check_boto_reqs(check_boto3=False)
    if has_boto_requirements is True:
        global __salt__
        if not __salt__:
            __salt__ = minion_mods(__opts__)
        return __virtualname__
    return has_boto_requirements

def _get_profile(service, region, key, keyid, profile):
    if False:
        return 10
    if profile:
        if isinstance(profile, str):
            _profile = __salt__['config.option'](profile)
        elif isinstance(profile, dict):
            _profile = profile
        key = _profile.get('key', None)
        keyid = _profile.get('keyid', None)
        region = _profile.get('region', region or None)
    if not region and __salt__['config.option'](service + '.region'):
        region = __salt__['config.option'](service + '.region')
    if not region:
        region = 'us-east-1'
    if not key and __salt__['config.option'](service + '.key'):
        key = __salt__['config.option'](service + '.key')
    if not keyid and __salt__['config.option'](service + '.keyid'):
        keyid = __salt__['config.option'](service + '.keyid')
    label = 'boto_{}:'.format(service)
    if keyid:
        hash_string = region + keyid + key
        hash_string = salt.utils.stringutils.to_bytes(hash_string)
        cxkey = label + hashlib.md5(hash_string).hexdigest()
    else:
        cxkey = label + region
    return (cxkey, region, key, keyid)

def cache_id(service, name, sub_resource=None, resource_id=None, invalidate=False, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Cache, invalidate, or retrieve an AWS resource id keyed by name.\n\n    .. code-block:: python\n\n        __utils__['boto.cache_id']('ec2', 'myinstance',\n                                   'i-a1b2c3',\n                                   profile='custom_profile')\n    "
    (cxkey, _, _, _) = _get_profile(service, region, key, keyid, profile)
    if sub_resource:
        cxkey = '{}:{}:{}:id'.format(cxkey, sub_resource, name)
    else:
        cxkey = '{}:{}:id'.format(cxkey, name)
    if invalidate:
        if cxkey in __context__:
            del __context__[cxkey]
            return True
        elif resource_id in __context__.values():
            ctx = {k: v for (k, v) in __context__.items() if v != resource_id}
            __context__.clear()
            __context__.update(ctx)
            return True
        else:
            return False
    if resource_id:
        __context__[cxkey] = resource_id
        return True
    return __context__.get(cxkey)

def cache_id_func(service):
    if False:
        print('Hello World!')
    "\n    Returns a partial ``cache_id`` function for the provided service.\n\n    .. code-block:: python\n\n        cache_id = __utils__['boto.cache_id_func']('ec2')\n        cache_id('myinstance', 'i-a1b2c3')\n        instance_id = cache_id('myinstance')\n    "
    return partial(cache_id, service)

def get_connection(service, module=None, region=None, key=None, keyid=None, profile=None):
    if False:
        print('Hello World!')
    "\n    Return a boto connection for the service.\n\n    .. code-block:: python\n\n        conn = __utils__['boto.get_connection']('ec2', profile='custom_profile')\n    "
    module = str(module or service)
    (module, submodule) = ('boto.' + module).rsplit('.', 1)
    svc_mod = getattr(__import__(module, fromlist=[submodule]), submodule)
    (cxkey, region, key, keyid) = _get_profile(service, region, key, keyid, profile)
    cxkey = cxkey + ':conn'
    if cxkey in __context__:
        return __context__[cxkey]
    try:
        conn = svc_mod.connect_to_region(region, aws_access_key_id=keyid, aws_secret_access_key=key)
        if conn is None:
            raise SaltInvocationError('Region "{}" is not valid.'.format(region))
    except boto.exception.NoAuthHandlerFound:
        raise SaltInvocationError('No authentication credentials found when attempting to make boto {} connection to region "{}".'.format(service, region))
    __context__[cxkey] = conn
    return conn

def get_connection_func(service, module=None):
    if False:
        return 10
    "\n    Returns a partial ``get_connection`` function for the provided service.\n\n    .. code-block:: python\n\n        get_conn = __utils__['boto.get_connection_func']('ec2')\n        conn = get_conn()\n    "
    return partial(get_connection, service, module=module)

def get_error(e):
    if False:
        while True:
            i = 10
    aws = {}
    if hasattr(e, 'status'):
        aws['status'] = e.status
    if hasattr(e, 'reason'):
        aws['reason'] = e.reason
    if hasattr(e, 'message') and e.message != '':
        aws['message'] = e.message
    if hasattr(e, 'error_code') and e.error_code is not None:
        aws['code'] = e.error_code
    if 'message' in aws and 'reason' in aws:
        message = '{}: {}'.format(aws['reason'], aws['message'])
    elif 'message' in aws:
        message = aws['message']
    elif 'reason' in aws:
        message = aws['reason']
    else:
        message = ''
    r = {'message': message}
    if aws:
        r['aws'] = aws
    return r

def exactly_n(l, n=1):
    if False:
        print('Hello World!')
    '\n    Tests that exactly N items in an iterable are "truthy" (neither None,\n    False, nor 0).\n    '
    i = iter(l)
    return all((any(i) for j in range(n))) and (not any(i))

def exactly_one(l):
    if False:
        i = 10
        return i + 15
    return exactly_n(l)

def assign_funcs(modname, service, module=None, pack=None):
    if False:
        print('Hello World!')
    "\n    Assign _get_conn and _cache_id functions to the named module.\n\n    .. code-block:: python\n\n        __utils__['boto.assign_partials'](__name__, 'ec2')\n    "
    if pack:
        global __salt__
        __salt__ = pack
    mod = sys.modules[modname]
    setattr(mod, '_get_conn', get_connection_func(service, module=module))
    setattr(mod, '_cache_id', cache_id_func(service))
    setattr(mod, '_exactly_one', exactly_one)

def paged_call(function, *args, **kwargs):
    if False:
        return 10
    '\n    Retrieve full set of values from a boto API call that may truncate\n    its results, yielding each page as it is obtained.\n    '
    marker_flag = kwargs.pop('marker_flag', 'marker')
    marker_arg = kwargs.pop('marker_flag', 'marker')
    while True:
        ret = function(*args, **kwargs)
        marker = ret.get(marker_flag)
        yield ret
        if not marker:
            break
        kwargs[marker_arg] = marker