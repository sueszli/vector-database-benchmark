"""
Boto3 Common Utils
=================

Note: This module depends on the dicts packed by the loader and,
therefore, must be accessed via the loader or from the __utils__ dict.

This module provides common functionality for the boto execution modules.
The expected usage is to call `apply_funcs` from the `__virtual__` function
of the module. This will bring properly initilized partials of  `_get_conn`
and `_cache_id` into the module's namespace.

Example Usage:

    .. code-block:: python

        def __virtual__():
            __utils__['boto.apply_funcs'](__name__, 'vpc')

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
try:
    import boto3
    import boto3.session
    import botocore
    import botocore.exceptions
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False
log = logging.getLogger(__name__)
__virtualname__ = 'boto3'
__salt_loader__ = salt.loader.context.LoaderContext()
__context__ = __salt_loader__.named_context('__context__', {})

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if boto libraries exist and if boto libraries are greater than\n    a given version.\n    '
    has_boto = salt.utils.versions.check_boto_reqs()
    if has_boto is True:
        return __virtualname__
    return has_boto

def _option(value):
    if False:
        print('Hello World!')
    '\n    Look up the value for an option.\n    '
    if value in __opts__:
        return __opts__[value]
    master_opts = __pillar__.get('master', {})
    if value in master_opts:
        return master_opts[value]
    if value in __pillar__:
        return __pillar__[value]

def _get_profile(service, region, key, keyid, profile):
    if False:
        print('Hello World!')
    if profile:
        if isinstance(profile, str):
            _profile = _option(profile)
        elif isinstance(profile, dict):
            _profile = profile
        key = _profile.get('key', None)
        keyid = _profile.get('keyid', None)
        region = _profile.get('region', None)
    if not region and _option(service + '.region'):
        region = _option(service + '.region')
    if not region:
        region = 'us-east-1'
        log.info('Assuming default region %s', region)
    if not key and _option(service + '.key'):
        key = _option(service + '.key')
    if not keyid and _option(service + '.keyid'):
        keyid = _option(service + '.keyid')
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
    "\n    Returns a partial `cache_id` function for the provided service.\n\n    .. code-block:: python\n\n        cache_id = __utils__['boto.cache_id_func']('ec2')\n        cache_id('myinstance', 'i-a1b2c3')\n        instance_id = cache_id('myinstance')\n    "
    return partial(cache_id, service)

def get_connection(service, module=None, region=None, key=None, keyid=None, profile=None):
    if False:
        while True:
            i = 10
    "\n    Return a boto connection for the service.\n\n    .. code-block:: python\n\n        conn = __utils__['boto.get_connection']('ec2', profile='custom_profile')\n    "
    module = module or service
    (cxkey, region, key, keyid) = _get_profile(service, region, key, keyid, profile)
    cxkey = cxkey + ':conn3'
    if cxkey in __context__:
        return __context__[cxkey]
    try:
        session = boto3.session.Session(aws_access_key_id=keyid, aws_secret_access_key=key, region_name=region)
        if session is None:
            raise SaltInvocationError('Region "{}" is not valid.'.format(region))
        conn = session.client(module)
        if conn is None:
            raise SaltInvocationError('Region "{}" is not valid.'.format(region))
    except botocore.exceptions.NoCredentialsError:
        raise SaltInvocationError('No authentication credentials found when attempting to make boto {} connection to region "{}".'.format(service, region))
    __context__[cxkey] = conn
    return conn

def get_connection_func(service, module=None):
    if False:
        while True:
            i = 10
    "\n    Returns a partial `get_connection` function for the provided service.\n\n    .. code-block:: python\n\n        get_conn = __utils__['boto.get_connection_func']('ec2')\n        conn = get_conn()\n    "
    return partial(get_connection, service, module=module)

def get_region(service, region, profile):
    if False:
        i = 10
        return i + 15
    '\n    Retrieve the region for a particular AWS service based on configured region and/or profile.\n    '
    (_, region, _, _) = _get_profile(service, region, None, None, profile)
    return region

def get_error(e):
    if False:
        for i in range(10):
            print('nop')
    aws = {}
    message = ''
    message = e.args[0]
    r = {'message': message}
    if aws:
        r['aws'] = aws
    return r

def exactly_n(l, n=1):
    if False:
        return 10
    '\n    Tests that exactly N items in an iterable are "truthy" (neither None,\n    False, nor 0).\n    '
    i = iter(l)
    return all((any(i) for j in range(n))) and (not any(i))

def exactly_one(l):
    if False:
        return 10
    return exactly_n(l)

def assign_funcs(modname, service, module=None, get_conn_funcname='_get_conn', cache_id_funcname='_cache_id', exactly_one_funcname='_exactly_one'):
    if False:
        i = 10
        return i + 15
    "\n    Assign _get_conn and _cache_id functions to the named module.\n\n    .. code-block:: python\n\n        _utils__['boto.assign_partials'](__name__, 'ec2')\n    "
    mod = sys.modules[modname]
    setattr(mod, get_conn_funcname, get_connection_func(service, module=module))
    setattr(mod, cache_id_funcname, cache_id_func(service))
    if exactly_one_funcname is not None:
        setattr(mod, exactly_one_funcname, exactly_one)

def paged_call(function, *args, **kwargs):
    if False:
        return 10
    'Retrieve full set of values from a boto3 API call that may truncate\n    its results, yielding each page as it is obtained.\n    '
    marker_flag = kwargs.pop('marker_flag', 'NextMarker')
    marker_arg = kwargs.pop('marker_arg', 'Marker')
    while True:
        ret = function(*args, **kwargs)
        marker = ret.get(marker_flag)
        yield ret
        if not marker:
            break
        kwargs[marker_arg] = marker

def ordered(obj):
    if False:
        print('Hello World!')
    if isinstance(obj, (list, tuple)):
        return sorted((ordered(x) for x in obj))
    elif isinstance(obj, dict):
        return {str(k) if isinstance(k, str) else k: ordered(v) for (k, v) in obj.items()}
    elif isinstance(obj, str):
        return str(obj)
    return obj

def json_objs_equal(left, right):
    if False:
        i = 10
        return i + 15
    'Compare two parsed JSON objects, given non-ordering in JSON objects'
    return ordered(left) == ordered(right)