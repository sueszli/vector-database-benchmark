"""
Glance module for interacting with OpenStack Glance

.. versionadded:: 2018.3.0

:depends:shade

Example configuration

.. code-block:: yaml

    glance:
      cloud: default

.. code-block:: yaml

    glance:
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
__virtualname__ = 'glanceng'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load this module if shade python module is installed\n    '
    if HAS_SHADE:
        return __virtualname__
    return (False, 'The glanceng execution module failed to load: shade python module is not available')

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

def _clean_kwargs(keep_name=False, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Sanatize the arguments for use with shade\n    '
    if 'name' in kwargs and (not keep_name):
        kwargs['name_or_id'] = kwargs.pop('name')
    return __utils__['args.clean_kwargs'](**kwargs)

def setup_clouds(auth=None):
    if False:
        while True:
            i = 10
    "\n    Call functions to create Shade cloud objects in __context__ to take\n    advantage of Shade's in-memory caching across several states\n    "
    get_operator_cloud(auth)
    get_openstack_cloud(auth)

def get_operator_cloud(auth=None):
    if False:
        print('Hello World!')
    '\n    Return an operator_cloud\n    '
    if auth is None:
        auth = __salt__['config.option']('glance', {})
    if 'shade_opcloud' in __context__:
        if __context__['shade_opcloud'].auth == auth:
            return __context__['shade_opcloud']
    __context__['shade_opcloud'] = shade.operator_cloud(**auth)
    return __context__['shade_opcloud']

def get_openstack_cloud(auth=None):
    if False:
        print('Hello World!')
    '\n    Return an openstack_cloud\n    '
    if auth is None:
        auth = __salt__['config.option']('glance', {})
    if 'shade_oscloud' in __context__:
        if __context__['shade_oscloud'].auth == auth:
            return __context__['shade_oscloud']
    __context__['shade_oscloud'] = shade.openstack_cloud(**auth)
    return __context__['shade_oscloud']

def image_create(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Create an image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glanceng.image_create name=cirros file=cirros.raw disk_format=raw\n        salt '*' glanceng.image_create name=cirros file=cirros.raw disk_format=raw hw_scsi_model=virtio-scsi hw_disk_bus=scsi\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(keep_name=True, **kwargs)
    return cloud.create_image(**kwargs)

def image_delete(auth=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Delete an image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glanceng.image_delete name=image1\n        salt '*' glanceng.image_delete name=0e4febc2a5ab4f2c8f374b054162506d\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.delete_image(**kwargs)

def image_list(auth=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    List images\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glanceng.image_list\n        salt '*' glanceng.image_list\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.list_images(**kwargs)

def image_search(auth=None, **kwargs):
    if False:
        return 10
    "\n    Search for images\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glanceng.image_search name=image1\n        salt '*' glanceng.image_search\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.search_images(**kwargs)

def image_get(auth=None, **kwargs):
    if False:
        return 10
    "\n    Get a single image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glanceng.image_get name=image1\n        salt '*' glanceng.image_get name=0e4febc2a5ab4f2c8f374b054162506d\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.get_image(**kwargs)

def update_image_properties(auth=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Update properties for an image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glanceng.update_image_properties name=image1 hw_scsi_model=virtio-scsi hw_disk_bus=scsi\n        salt '*' glanceng.update_image_properties name=0e4febc2a5ab4f2c8f374b054162506d min_ram=1024\n    "
    cloud = get_operator_cloud(auth)
    kwargs = _clean_kwargs(**kwargs)
    return cloud.update_image_properties(**kwargs)