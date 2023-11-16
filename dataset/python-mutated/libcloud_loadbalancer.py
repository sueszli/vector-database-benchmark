"""
Apache Libcloud Load Balancer Management
========================================

Connection module for Apache Libcloud Storage load balancer management for a full list
of supported clouds, see http://libcloud.readthedocs.io/en/latest/loadbalancer/supported_providers.html

Clouds include Amazon ELB, ALB, Google, Aliyun, CloudStack, Softlayer

.. versionadded:: 2018.3.0

:configuration:
    This module uses a configuration profile for one or multiple Storage providers

    .. code-block:: yaml

        libcloud_loadbalancer:
            profile_test1:
              driver: gce
              key: GOOG0123456789ABCXYZ
              secret: mysecret
            profile_test2:
              driver: alb
              key: 12345
              secret: mysecret

:depends: apache-libcloud
"""
import logging
import salt.utils.args
import salt.utils.compat
from salt.utils.versions import Version
log = logging.getLogger(__name__)
REQUIRED_LIBCLOUD_VERSION = '1.5.0'
try:
    import libcloud
    from libcloud.loadbalancer.base import Algorithm, Member
    from libcloud.loadbalancer.providers import get_driver
    if hasattr(libcloud, '__version__') and Version(libcloud.__version__) < Version(REQUIRED_LIBCLOUD_VERSION):
        raise ImportError()
    logging.getLogger('libcloud').setLevel(logging.CRITICAL)
    HAS_LIBCLOUD = True
except ImportError:
    HAS_LIBCLOUD = False

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if libcloud libraries exist.\n    '
    if not HAS_LIBCLOUD:
        return (False, 'A apache-libcloud library with version at least {} was not found'.format(REQUIRED_LIBCLOUD_VERSION))
    return True

def _algorithm_maps():
    if False:
        return 10
    return {'RANDOM': Algorithm.RANDOM, 'ROUND_ROBIN': Algorithm.ROUND_ROBIN, 'LEAST_CONNECTIONS': Algorithm.LEAST_CONNECTIONS, 'WEIGHTED_ROUND_ROBIN': Algorithm.WEIGHTED_ROUND_ROBIN, 'WEIGHTED_LEAST_CONNECTIONS': Algorithm.WEIGHTED_LEAST_CONNECTIONS, 'SHORTEST_RESPONSE': Algorithm.SHORTEST_RESPONSE, 'PERSISTENT_IP': Algorithm.PERSISTENT_IP}

def _get_driver(profile):
    if False:
        for i in range(10):
            print('nop')
    config = __salt__['config.option']('libcloud_loadbalancer')[profile]
    cls = get_driver(config['driver'])
    args = config.copy()
    del args['driver']
    args['key'] = config.get('key')
    args['secret'] = config.get('secret', None)
    if args['secret'] is None:
        del args['secret']
    args['secure'] = config.get('secure', True)
    args['host'] = config.get('host', None)
    args['port'] = config.get('port', None)
    return cls(**args)

def list_balancers(profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of load balancers.\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_balancers method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.list_balancers profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    balancers = conn.list_balancers(**libcloud_kwargs)
    ret = []
    for balancer in balancers:
        ret.append(_simple_balancer(balancer))
    return ret

def list_protocols(profile, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Return a list of supported protocols.\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_protocols method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: a list of supported protocols\n    :rtype: ``list`` of ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.list_protocols profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    return conn.list_protocols(**libcloud_kwargs)

def create_balancer(name, port, protocol, profile, algorithm=None, members=None, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a new load balancer instance\n\n    :param name: Name of the new load balancer (required)\n    :type  name: ``str``\n\n    :param port: Port the load balancer should listen on, defaults to 80\n    :type  port: ``str``\n\n    :param protocol: Loadbalancer protocol, defaults to http.\n    :type  protocol: ``str``\n\n    :param algorithm: Load balancing algorithm, defaults to ROUND_ROBIN. See Algorithm type\n        in Libcloud documentation for a full listing.\n    :type algorithm: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's create_balancer method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: The details of the new balancer\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.create_balancer my_balancer 80 http profile1\n    "
    if algorithm is None:
        algorithm = Algorithm.ROUND_ROBIN
    elif isinstance(algorithm, str):
        algorithm = _algorithm_maps()[algorithm]
    starting_members = []
    if members is not None:
        if isinstance(members, list):
            for m in members:
                starting_members.append(Member(id=None, ip=m['ip'], port=m['port']))
        else:
            raise ValueError('members must be of type list')
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    conn = _get_driver(profile=profile)
    balancer = conn.create_balancer(name, port, protocol, algorithm, starting_members, **libcloud_kwargs)
    return _simple_balancer(balancer)

def destroy_balancer(balancer_id, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Destroy a load balancer\n\n    :param balancer_id: LoadBalancer ID which should be used\n    :type  balancer_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's destroy_balancer method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: ``True`` if the destroy was successful, otherwise ``False``.\n    :rtype: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.destroy_balancer balancer_1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    balancer = conn.get_balancer(balancer_id)
    return conn.destroy_balancer(balancer, **libcloud_kwargs)

def get_balancer_by_name(name, profile, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Get the details for a load balancer by name\n\n    :param name: Name of a load balancer you want to fetch\n    :type  name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_balancers method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: the load balancer details\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.get_balancer_by_name my_balancer profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    balancers = conn.list_balancers(**libcloud_kwargs)
    match = [b for b in balancers if b.name == name]
    if len(match) == 1:
        return _simple_balancer(match[0])
    elif len(match) > 1:
        raise ValueError('Ambiguous argument, found mulitple records')
    else:
        raise ValueError('Bad argument, found no records')

def get_balancer(balancer_id, profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the details for a load balancer by ID\n\n    :param balancer_id: id of a load balancer you want to fetch\n    :type  balancer_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's get_balancer method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: the load balancer details\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.get_balancer balancer123 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    balancer = conn.get_balancer(balancer_id, **libcloud_kwargs)
    return _simple_balancer(balancer)

def list_supported_algorithms(profile, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Get the supported algorithms for a profile\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_supported_algorithms method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: The supported algorithms\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.list_supported_algorithms profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    return conn.list_supported_algorithms(**libcloud_kwargs)

def balancer_attach_member(balancer_id, ip, port, profile, extra=None, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add a new member to the load balancer\n\n    :param balancer_id: id of a load balancer you want to fetch\n    :type  balancer_id: ``str``\n\n    :param ip: IP address for the new member\n    :type  ip: ``str``\n\n    :param port: Port for the new member\n    :type  port: ``int``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's balancer_attach_member method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.balancer_attach_member balancer123 1.2.3.4 80 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    member = Member(id=None, ip=ip, port=port, balancer=None, extra=extra)
    balancer = conn.get_balancer(balancer_id)
    member_saved = conn.balancer_attach_member(balancer, member, **libcloud_kwargs)
    return _simple_member(member_saved)

def balancer_detach_member(balancer_id, member_id, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Add a new member to the load balancer\n\n    :param balancer_id: id of a load balancer you want to fetch\n    :type  balancer_id: ``str``\n\n    :param ip: IP address for the new member\n    :type  ip: ``str``\n\n    :param port: Port for the new member\n    :type  port: ``int``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's balancer_detach_member method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.balancer_detach_member balancer123 member123 profile1\n    "
    conn = _get_driver(profile=profile)
    balancer = conn.get_balancer(balancer_id)
    members = conn.balancer_list_members(balancer=balancer)
    match = [member for member in members if member.id == member_id]
    if len(match) > 1:
        raise ValueError('Ambiguous argument, found mulitple records')
    elif not match:
        raise ValueError('Bad argument, found no records')
    else:
        member = match[0]
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    return conn.balancer_detach_member(balancer=balancer, member=member, **libcloud_kwargs)

def list_balancer_members(balancer_id, profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    List the members of a load balancer\n\n    :param balancer_id: id of a load balancer you want to fetch\n    :type  balancer_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_balancer_members method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.list_balancer_members balancer123 profile1\n    "
    conn = _get_driver(profile=profile)
    balancer = conn.get_balancer(balancer_id)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    members = conn.balancer_list_members(balancer=balancer, **libcloud_kwargs)
    return [_simple_member(member) for member in members]

def extra(method, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Call an extended method on the driver\n\n    :param method: Driver's method name\n    :type  method: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_loadbalancer.extra ex_get_permissions google container_name=my_container object_name=me.jpg --out=yaml\n    "
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    conn = _get_driver(profile=profile)
    connection_method = getattr(conn, method)
    return connection_method(**libcloud_kwargs)

def _simple_balancer(balancer):
    if False:
        return 10
    return {'id': balancer.id, 'name': balancer.name, 'state': balancer.state, 'ip': balancer.ip, 'port': balancer.port, 'extra': balancer.extra}

def _simple_member(member):
    if False:
        return 10
    return {'id': member.id, 'ip': member.ip, 'port': member.port, 'balancer': _simple_balancer(member.balancer), 'extra': member.extra}