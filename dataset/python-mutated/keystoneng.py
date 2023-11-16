"""
Keystone module for interacting with OpenStack Keystone

.. versionadded:: 2018.3.0

:depends:shade

Example configuration

.. code-block:: yaml

    keystone:
      cloud: default

.. code-block:: yaml

    keystone:
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
    from shade.exc import OpenStackCloudException
    HAS_SHADE = True
except ImportError:
    pass
__virtualname__ = 'keystoneng'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load this module if shade python module is installed\n    '
    if HAS_SHADE:
        return __virtualname__
    return (False, 'The keystoneng execution module failed to load: shade python module is not available')

def compare_changes(obj, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compare two dicts returning only keys that exist in the first dict and are\n    different in the second one\n    '
    changes = {}
    for (k, v) in obj.items():
        if k in kwargs:
            if v != kwargs[k]:
                changes[k] = kwargs[k]
    return changes

def get_entity(ent_type, **kwargs):
    if False:
        return 10
    '\n    Attempt to query Keystone for more information about an entity\n    '
    try:
        func = 'keystoneng.{}_get'.format(ent_type)
        ent = __salt__[func](**kwargs)
    except OpenStackCloudException as e:
        if 'HTTP 403' not in e.inner_exception[1][0]:
            raise
        ent = kwargs['name']
    return ent

def _clean_kwargs(keep_name=False, **kwargs):
    if False:
        return 10
    '\n    Sanatize the arguments for use with shade\n    '
    if 'name' in kwargs and (not keep_name):
        kwargs['name_or_id'] = kwargs.pop('name')
    return __utils__['args.clean_kwargs'](**kwargs)

def setup_clouds(auth=None):
    if False:
        print('Hello World!')
    "\n    Call functions to create Shade cloud objects in __context__ to take\n    advantage of Shade's in-memory caching across several states\n    "
    get_operator_cloud(auth)
    get_openstack_cloud(auth)

def get_operator_cloud(auth=None):
    if False:
        print('Hello World!')
    '\n    Return an operator_cloud\n    '
    if auth is None:
        auth = __salt__['config.option']('keystone', {})
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
        auth = __salt__['config.option']('keystone', {})
    if 'shade_oscloud' in __context__:
        if __context__['shade_oscloud'].auth == auth:
            return __context__['shade_oscloud']
    __context__['shade_oscloud'] = shade.openstack_cloud(**auth)
    return __context__['shade_oscloud']

def group_create(auth=None, **kwargs):
    if False:
        return 10
    "\n    Create a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.group_create name=group1\n        salt '*' keystoneng.group_create name=group2 domain=domain1 description='my group2'\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_group(**kwargs)

def group_delete(auth=None, **kwargs):
    if False:
        return 10
    "\n    Delete a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.group_delete name=group1\n        salt '*' keystoneng.group_delete name=group2 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.group_delete name=0e4febc2a5ab4f2c8f374b054162506d\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_group(**kwargs)

def group_update(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Update a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.group_update name=group1 description='new description'\n        salt '*' keystoneng.group_create name=group2 domain_id=b62e76fbeeff4e8fb77073f591cf211e new_name=newgroupname\n        salt '*' keystoneng.group_create name=0e4febc2a5ab4f2c8f374b054162506d new_name=newgroupname\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    if 'new_name' in kwargs:
        kwargs['name'] = kwargs.pop('new_name')
    return cloud.update_group(**kwargs)

def group_list(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List groups\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.group_list\n        salt '*' keystoneng.group_list domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_groups(**kwargs)

def group_search(auth=None, **kwargs):
    if False:
        return 10
    "\n    Search for groups\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.group_search name=group1\n        salt '*' keystoneng.group_search domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_groups(**kwargs)

def group_get(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Get a single group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.group_get name=group1\n        salt '*' keystoneng.group_get name=group2 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.group_get name=0e4febc2a5ab4f2c8f374b054162506d\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_group(**kwargs)

def project_create(auth=None, **kwargs):
    if False:
        return 10
    "\n    Create a project\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.project_create name=project1\n        salt '*' keystoneng.project_create name=project2 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.project_create name=project3 enabled=False description='my project3'\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_project(**kwargs)

def project_delete(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Delete a project\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.project_delete name=project1\n        salt '*' keystoneng.project_delete name=project2 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.project_delete name=f315afcf12f24ad88c92b936c38f2d5a\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_project(**kwargs)

def project_update(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Update a project\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.project_update name=project1 new_name=newproject\n        salt '*' keystoneng.project_update name=project2 enabled=False description='new description'\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    if 'new_name' in kwargs:
        kwargs['name'] = kwargs.pop('new_name')
    return cloud.update_project(**kwargs)

def project_list(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List projects\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.project_list\n        salt '*' keystoneng.project_list domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_projects(**kwargs)

def project_search(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Search projects\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.project_search\n        salt '*' keystoneng.project_search name=project1\n        salt '*' keystoneng.project_search domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_projects(**kwargs)

def project_get(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Get a single project\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.project_get name=project1\n        salt '*' keystoneng.project_get name=project2 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.project_get name=f315afcf12f24ad88c92b936c38f2d5a\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_project(**kwargs)

def domain_create(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.domain_create name=domain1\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_domain(**kwargs)

def domain_delete(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.domain_delete name=domain1\n        salt '*' keystoneng.domain_delete name=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_domain(**kwargs)

def domain_update(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Update a domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.domain_update name=domain1 new_name=newdomain\n        salt '*' keystoneng.domain_update name=domain1 enabled=True description='new description'\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    if 'new_name' in kwargs:
        kwargs['name'] = kwargs.pop('new_name')
    return cloud.update_domain(**kwargs)

def domain_list(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    List domains\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.domain_list\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_domains(**kwargs)

def domain_search(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Search domains\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.domain_search\n        salt '*' keystoneng.domain_search name=domain1\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_domains(**kwargs)

def domain_get(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Get a single domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.domain_get name=domain1\n        salt '*' keystoneng.domain_get name=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_domain(**kwargs)

def role_create(auth=None, **kwargs):
    if False:
        return 10
    "\n    Create a role\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_create name=role1\n        salt '*' keystoneng.role_create name=role1 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_role(**kwargs)

def role_delete(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a role\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_delete name=role1 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.role_delete name=1eb6edd5525e4ac39af571adee673559\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_role(**kwargs)

def role_update(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update a role\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_update name=role1 new_name=newrole\n        salt '*' keystoneng.role_update name=1eb6edd5525e4ac39af571adee673559 new_name=newrole\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    if 'new_name' in kwargs:
        kwargs['name'] = kwargs.pop('new_name')
    return cloud.update_role(**kwargs)

def role_list(auth=None, **kwargs):
    if False:
        return 10
    "\n    List roles\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_list\n        salt '*' keystoneng.role_list domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_roles(**kwargs)

def role_search(auth=None, **kwargs):
    if False:
        return 10
    "\n    Search roles\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_search\n        salt '*' keystoneng.role_search name=role1\n        salt '*' keystoneng.role_search domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_roles(**kwargs)

def role_get(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get a single role\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_get name=role1\n        salt '*' keystoneng.role_get name=role1 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.role_get name=1eb6edd5525e4ac39af571adee673559\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_role(**kwargs)

def user_create(auth=None, **kwargs):
    if False:
        return 10
    "\n    Create a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.user_create name=user1\n        salt '*' keystoneng.user_create name=user2 password=1234 enabled=False\n        salt '*' keystoneng.user_create name=user3 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_user(**kwargs)

def user_delete(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Delete a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.user_delete name=user1\n        salt '*' keystoneng.user_delete name=user2 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.user_delete name=a42cbbfa1e894e839fd0f584d22e321f\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_user(**kwargs)

def user_update(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.user_update name=user1 enabled=False description='new description'\n        salt '*' keystoneng.user_update name=user1 new_name=newuser\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    if 'new_name' in kwargs:
        kwargs['name'] = kwargs.pop('new_name')
    return cloud.update_user(**kwargs)

def user_list(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.user_list\n        salt '*' keystoneng.user_list domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_users(**kwargs)

def user_search(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.user_list\n        salt '*' keystoneng.user_list domain_id=b62e76fbeeff4e8fb77073f591cf211e\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_users(**kwargs)

def user_get(auth=None, **kwargs):
    if False:
        return 10
    "\n    Get a single user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.user_get name=user1\n        salt '*' keystoneng.user_get name=user1 domain_id=b62e76fbeeff4e8fb77073f591cf211e\n        salt '*' keystoneng.user_get name=02cffaa173b2460f98e40eda3748dae5\n    "
    cloud = get_openstack_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_user(**kwargs)

def endpoint_create(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Create an endpoint\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.endpoint_create interface=admin service=glance url=https://example.org:9292\n        salt '*' keystoneng.endpoint_create interface=public service=glance region=RegionOne url=https://example.org:9292\n        salt '*' keystoneng.endpoint_create interface=admin service=glance url=https://example.org:9292 enabled=True\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_endpoint(**kwargs)

def endpoint_delete(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Delete an endpoint\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.endpoint_delete id=3bee4bd8c2b040ee966adfda1f0bfca9\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_endpoint(**kwargs)

def endpoint_update(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update an endpoint\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.endpoint_update endpoint_id=4f961ad09d2d48948896bbe7c6a79717 interface=public enabled=False\n        salt '*' keystoneng.endpoint_update endpoint_id=4f961ad09d2d48948896bbe7c6a79717 region=newregion\n        salt '*' keystoneng.endpoint_update endpoint_id=4f961ad09d2d48948896bbe7c6a79717 service_name_or_id=glance url=https://example.org:9292\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.update_endpoint(**kwargs)

def endpoint_list(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List endpoints\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.endpoint_list\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_endpoints(**kwargs)

def endpoint_search(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Search endpoints\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.endpoint_search\n        salt '*' keystoneng.endpoint_search id=02cffaa173b2460f98e40eda3748dae5\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_endpoints(**kwargs)

def endpoint_get(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Get a single endpoint\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.endpoint_get id=02cffaa173b2460f98e40eda3748dae5\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_endpoint(**kwargs)

def service_create(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' keystoneng.service_create name=glance type=image\n        salt \'*\' keystoneng.service_create name=glance type=image description="Image"\n    '
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_service(**kwargs)

def service_delete(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Delete a service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.service_delete name=glance\n        salt '*' keystoneng.service_delete name=39cc1327cdf744ab815331554430e8ec\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_service(**kwargs)

def service_update(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Update a service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.service_update name=cinder type=volumev2\n        salt '*' keystoneng.service_update name=cinder description='new description'\n        salt '*' keystoneng.service_update name=ab4d35e269f147b3ae2d849f77f5c88f enabled=False\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.update_service(**kwargs)

def service_list(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.service_list\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_services(**kwargs)

def service_search(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Search services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.service_search\n        salt '*' keystoneng.service_search name=glance\n        salt '*' keystoneng.service_search name=135f0403f8e544dc9008c6739ecda860\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_services(**kwargs)

def service_get(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Get a single service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.service_get name=glance\n        salt '*' keystoneng.service_get name=75a5804638944b3ab54f7fbfcec2305a\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_service(**kwargs)

def role_assignment_list(auth=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List role assignments\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_assignment_list\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_role_assignments(**kwargs)

def role_grant(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Grant a role in a project/domain to a user/group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_grant name=role1 user=user1 project=project1\n        salt '*' keystoneng.role_grant name=ddbe3e0ed74e4c7f8027bad4af03339d group=user1 project=project1 domain=domain1\n        salt '*' keystoneng.role_grant name=ddbe3e0ed74e4c7f8027bad4af03339d group=19573afd5e4241d8b65c42215bae9704 project=1dcac318a83b4610b7a7f7ba01465548\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.grant_role(**kwargs)

def role_revoke(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Grant a role in a project/domain to a user/group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keystoneng.role_revoke name=role1 user=user1 project=project1\n        salt '*' keystoneng.role_revoke name=ddbe3e0ed74e4c7f8027bad4af03339d group=user1 project=project1 domain=domain1\n        salt '*' keystoneng.role_revoke name=ddbe3e0ed74e4c7f8027bad4af03339d group=19573afd5e4241d8b65c42215bae9704 project=1dcac318a83b4610b7a7f7ba01465548\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.revoke_role(**kwargs)