"""
Neutron module for interacting with OpenStack Neutron

.. versionadded:: 2018.3.0

:depends:shade

Example configuration

.. code-block:: yaml

    neutron:
      cloud: default

.. code-block:: yaml

    neutron:
      auth:
        username: admin
        password: password123
        user_domain_name: mydomain
        project_name: myproject
        project_domain_name: myproject
        auth_url: https://example.org:5000/v3
      identity_api_version: 3
"""
HAS_SHADE = False
try:
    import shade
    HAS_SHADE = True
except ImportError:
    pass
__virtualname__ = 'neutronng'

def __virtual__():
    if False:
        return 10
    '\n    Only load this module if shade python module is installed\n    '
    if HAS_SHADE:
        return __virtualname__
    return (False, 'The neutronng execution module failed to load: shade python module is not available')

def compare_changes(obj, **kwargs):
    if False:
        return 10
    '\n    Compare two dicts returning only keys that exist in the first dict and are\n    different in the second one\n    '
    changes = {}
    for (key, value) in obj.items():
        if key in kwargs:
            if value != kwargs[key]:
                changes[key] = kwargs[key]
    return changes

def _clean_kwargs(keep_name=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Sanatize the arguments for use with shade\n    '
    if 'name' in kwargs and (not keep_name):
        kwargs['name_or_id'] = kwargs.pop('name')
    return __utils__['args.clean_kwargs'](**kwargs)

def setup_clouds(auth=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Call functions to create Shade cloud objects in __context__ to take\n    advantage of Shade's in-memory caching across several states\n    "
    get_operator_cloud(auth)
    get_openstack_cloud(auth)

def get_operator_cloud(auth=None):
    if False:
        return 10
    '\n    Return an operator_cloud\n    '
    if auth is None:
        auth = __salt__['config.option']('neutron', {})
    if 'shade_opcloud' in __context__:
        if __context__['shade_opcloud'].auth == auth:
            return __context__['shade_opcloud']
    __context__['shade_opcloud'] = shade.operator_cloud(**auth)
    return __context__['shade_opcloud']

def get_openstack_cloud(auth=None):
    if False:
        return 10
    '\n    Return an openstack_cloud\n    '
    if auth is None:
        auth = __salt__['config.option']('neutron', {})
    if 'shade_oscloud' in __context__:
        if __context__['shade_oscloud'].auth == auth:
            return __context__['shade_oscloud']
    __context__['shade_oscloud'] = shade.openstack_cloud(**auth)
    return __context__['shade_oscloud']

def network_create(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Create a network\n\n    name\n        Name of the network being created\n\n    shared : False\n        If ``True``, set the network as shared\n\n    admin_state_up : True\n        If ``True``, Set the network administrative state to "up"\n\n    external : False\n        Control whether or not this network is externally accessible\n\n    provider\n        An optional Python dictionary of network provider options\n\n    project_id\n        The project ID on which this network will be created\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.network_create name=network2           shared=True admin_state_up=True external=True\n\n        salt \'*\' neutronng.network_create name=network3           provider=\'{"network_type": "vlan",                     "segmentation_id": "4010",                     "physical_network": "provider"}\'           project_id=1dcac318a83b4610b7a7f7ba01465548\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_network(**kwargs)

def network_delete(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Delete a network\n\n    name_or_id\n        Name or ID of the network being deleted\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.network_delete name_or_id=network1\n        salt '*' neutronng.network_delete name_or_id=1dcac318a83b4610b7a7f7ba01465548\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_network(**kwargs)

def list_networks(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    List networks\n\n    filters\n        A Python dictionary of filter conditions to push down\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.list_networks\n        salt \'*\' neutronng.list_networks           filters=\'{"tenant_id": "1dcac318a83b4610b7a7f7ba01465548"}\'\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_networks(**kwargs)

def network_get(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get a single network\n\n    filters\n        A Python dictionary of filter conditions to push down\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.network_get name=XLB4\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_network(**kwargs)

def subnet_create(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a subnet\n\n    network_name_or_id\n        The unique name or ID of the attached network. If a non-unique name is\n        supplied, an exception is raised.\n\n    cidr\n        The CIDR\n\n    ip_version\n        The IP version, which is 4 or 6.\n\n    enable_dhcp : False\n        Set to ``True`` if DHCP is enabled and ``False`` if disabled\n\n    subnet_name\n        The name of the subnet\n\n    tenant_id\n        The ID of the tenant who owns the network. Only administrative users\n        can specify a tenant ID other than their own.\n\n    allocation_pools\n        A list of dictionaries of the start and end addresses for the\n        allocation pools.\n\n    gateway_ip\n        The gateway IP address. When you specify both ``allocation_pools`` and\n        ``gateway_ip``, you must ensure that the gateway IP does not overlap\n        with the specified allocation pools.\n\n    disable_gateway_ip : False\n        Set to ``True`` if gateway IP address is disabled and ``False`` if\n        enabled. It is not allowed with ``gateway_ip``.\n\n    dns_nameservers\n        A list of DNS name servers for the subnet\n\n    host_routes\n        A list of host route dictionaries for the subnet\n\n    ipv6_ra_mode\n        IPv6 Router Advertisement mode. Valid values are ``dhcpv6-stateful``,\n        ``dhcpv6-stateless``, or ``slaac``.\n\n    ipv6_address_mode\n        IPv6 address mode. Valid values are ``dhcpv6-stateful``,\n        ``dhcpv6-stateless``, or ``slaac``.\n\n    use_default_subnetpool\n        If ``True``, use the default subnetpool for ``ip_version`` to obtain a\n        CIDR. It is required to pass ``None`` to the ``cidr`` argument when\n        enabling this option.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.subnet_create network_name_or_id=network1\n          subnet_name=subnet1\n\n        salt \'*\' neutronng.subnet_create subnet_name=subnet2          network_name_or_id=network2 enable_dhcp=True           allocation_pools=\'[{"start": "192.168.199.2",                              "end": "192.168.199.254"}]\'          gateway_ip=\'192.168.199.1\' cidr=192.168.199.0/24\n\n        salt \'*\' neutronng.subnet_create network_name_or_id=network1           subnet_name=subnet1 dns_nameservers=\'["8.8.8.8", "8.8.8.7"]\'\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.create_subnet(**kwargs)

def subnet_update(auth=None, **kwargs):
    if False:
        return 10
    '\n    Update a subnet\n\n    name_or_id\n        Name or ID of the subnet to update\n\n    subnet_name\n        The new name of the subnet\n\n    enable_dhcp\n        Set to ``True`` if DHCP is enabled and ``False`` if disabled\n\n    gateway_ip\n        The gateway IP address. When you specify both allocation_pools and\n        gateway_ip, you must ensure that the gateway IP does not overlap with\n        the specified allocation pools.\n\n    disable_gateway_ip : False\n        Set to ``True`` if gateway IP address is disabled and False if enabled.\n        It is not allowed with ``gateway_ip``.\n\n    allocation_pools\n        A list of dictionaries of the start and end addresses for the\n        allocation pools.\n\n    dns_nameservers\n        A list of DNS name servers for the subnet\n\n    host_routes\n        A list of host route dictionaries for the subnet\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.subnet_update name=subnet1 subnet_name=subnet2\n        salt \'*\' neutronng.subnet_update name=subnet1 dns_nameservers=\'["8.8.8.8", "8.8.8.7"]\'\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.update_subnet(**kwargs)

def subnet_delete(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a subnet\n\n    name\n        Name or ID of the subnet to update\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.subnet_delete name=subnet1\n        salt '*' neutronng.subnet_delete           name=1dcac318a83b4610b7a7f7ba01465548\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_subnet(**kwargs)

def list_subnets(auth=None, **kwargs):
    if False:
        return 10
    '\n    List subnets\n\n    filters\n        A Python dictionary of filter conditions to push down\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.list_subnets\n        salt \'*\' neutronng.list_subnets           filters=\'{"tenant_id": "1dcac318a83b4610b7a7f7ba01465548"}\'\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_subnets(**kwargs)

def subnet_get(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Get a single subnet\n\n    filters\n        A Python dictionary of filter conditions to push down\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.subnet_get name=subnet1\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_subnet(**kwargs)

def security_group_create(auth=None, **kwargs):
    if False:
        return 10
    '\n    Create a security group. Use security_group_get to create default.\n\n    project_id\n        The project ID on which this security group will be created\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.security_group_create name=secgroup1           description="Very secure security group"\n        salt \'*\' neutronng.security_group_create name=secgroup1           description="Very secure security group"           project_id=1dcac318a83b4610b7a7f7ba01465548\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_security_group(**kwargs)

def security_group_update(secgroup=None, auth=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Update a security group\n\n    secgroup\n        Name, ID or Raw Object of the security group to update\n\n    name\n        New name for the security group\n\n    description\n        New description for the security group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.security_group_update secgroup=secgroup1           description="Very secure security group"\n        salt \'*\' neutronng.security_group_update secgroup=secgroup1           description="Very secure security group"           project_id=1dcac318a83b4610b7a7f7ba01465548\n\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.update_security_group(secgroup, **kwargs)

def security_group_delete(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Delete a security group\n\n    name_or_id\n        The name or unique ID of the security group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.security_group_delete name_or_id=secgroup1\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_security_group(**kwargs)

def security_group_get(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a single security group. This will create a default security group\n    if one does not exist yet for a particular project id.\n\n    filters\n        A Python dictionary of filter conditions to push down\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' neutronng.security_group_get           name=1dcac318a83b4610b7a7f7ba01465548\n\n        salt \'*\' neutronng.security_group_get           name=default          filters=\'{"tenant_id":"2e778bb64ca64a199eb526b5958d8710"}\'\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_security_group(**kwargs)

def security_group_rule_create(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a rule in a security group\n\n    secgroup_name_or_id\n        The security group name or ID to associate with this security group\n        rule. If a non-unique group name is given, an exception is raised.\n\n    port_range_min\n        The minimum port number in the range that is matched by the security\n        group rule. If the protocol is TCP or UDP, this value must be less than\n        or equal to the port_range_max attribute value. If nova is used by the\n        cloud provider for security groups, then a value of None will be\n        transformed to -1.\n\n    port_range_max\n        The maximum port number in the range that is matched by the security\n        group rule. The port_range_min attribute constrains the port_range_max\n        attribute. If nova is used by the cloud provider for security groups,\n        then a value of None will be transformed to -1.\n\n    protocol\n        The protocol that is matched by the security group rule.  Valid values\n        are ``None``, ``tcp``, ``udp``, and ``icmp``.\n\n    remote_ip_prefix\n        The remote IP prefix to be associated with this security group rule.\n        This attribute matches the specified IP prefix as the source IP address\n        of the IP packet.\n\n    remote_group_id\n        The remote group ID to be associated with this security group rule\n\n    direction\n        Either ``ingress`` or ``egress``; the direction in which the security\n        group rule is applied. For a compute instance, an ingress security\n        group rule is applied to incoming (ingress) traffic for that instance.\n        An egress rule is applied to traffic leaving the instance\n\n    ethertype\n        Must be IPv4 or IPv6, and addresses represented in CIDR must match the\n        ingress or egress rules\n\n    project_id\n        Specify the project ID this security group will be created on\n        (admin-only)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.security_group_rule_create          secgroup_name_or_id=secgroup1\n\n        salt '*' neutronng.security_group_rule_create          secgroup_name_or_id=secgroup2 port_range_min=8080          port_range_max=8080 direction='egress'\n\n        salt '*' neutronng.security_group_rule_create          secgroup_name_or_id=c0e1d1ce-7296-405e-919d-1c08217be529          protocol=icmp project_id=1dcac318a83b4610b7a7f7ba01465548\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.create_security_group_rule(**kwargs)

def security_group_rule_delete(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a security group\n\n    name_or_id\n        The unique ID of the security group rule\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' neutronng.security_group_rule_delete name_or_id=1dcac318a83b4610b7a7f7ba01465548\n\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_security_group_rule(**kwargs)