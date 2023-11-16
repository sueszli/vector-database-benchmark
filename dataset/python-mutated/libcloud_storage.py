"""
Apache Libcloud Storage Management
==================================

Connection module for Apache Libcloud Storage (object/blob) management for a full list
of supported clouds, see http://libcloud.readthedocs.io/en/latest/storage/supported_providers.html

Clouds include Amazon S3, Google Storage, Aliyun, Azure Blobs, Ceph, OpenStack swift

.. versionadded:: 2018.3.0

:configuration:
    This module uses a configuration profile for one or multiple Storage providers

    .. code-block:: yaml

        libcloud_storage:
            profile_test1:
              driver: google_storage
              key: GOOG0123456789ABCXYZ
              secret: mysecret
            profile_test2:
              driver: s3
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
    from libcloud.storage.providers import get_driver
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

def _get_driver(profile):
    if False:
        return 10
    config = __salt__['config.option']('libcloud_storage')[profile]
    cls = get_driver(config['driver'])
    args = config.copy()
    del args['driver']
    args['key'] = config.get('key')
    args['secret'] = config.get('secret', None)
    args['secure'] = config.get('secure', True)
    args['host'] = config.get('host', None)
    args['port'] = config.get('port', None)
    return cls(**args)

def list_containers(profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    Return a list of containers.\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_containers method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.list_containers profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    containers = conn.list_containers(**libcloud_kwargs)
    ret = []
    for container in containers:
        ret.append({'name': container.name, 'extra': container.extra})
    return ret

def list_container_objects(container_name, profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List container objects (e.g. files) for the given container_id on the given profile\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_container_objects method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.list_container_objects MyFolder profile1\n    "
    conn = _get_driver(profile=profile)
    container = conn.get_container(container_name)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    objects = conn.list_container_objects(container, **libcloud_kwargs)
    ret = []
    for obj in objects:
        ret.append({'name': obj.name, 'size': obj.size, 'hash': obj.hash, 'container': obj.container.name, 'extra': obj.extra, 'meta_data': obj.meta_data})
    return ret

def create_container(container_name, profile, **libcloud_kwargs):
    if False:
        print('Hello World!')
    "\n    Create a container in the cloud\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's create_container method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.create_container MyFolder profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    container = conn.create_container(container_name, **libcloud_kwargs)
    return {'name': container.name, 'extra': container.extra}

def get_container(container_name, profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List container details for the given container_name on the given profile\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's get_container method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.get_container MyFolder profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    container = conn.get_container(container_name, **libcloud_kwargs)
    return {'name': container.name, 'extra': container.extra}

def get_container_object(container_name, object_name, profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the details for a container object (file or object in the cloud)\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param object_name: Object name\n    :type  object_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's get_container_object method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.get_container_object MyFolder MyFile.xyz profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    obj = conn.get_container_object(container_name, object_name, **libcloud_kwargs)
    return {'name': obj.name, 'size': obj.size, 'hash': obj.hash, 'container': obj.container.name, 'extra': obj.extra, 'meta_data': obj.meta_data}

def download_object(container_name, object_name, destination_path, profile, overwrite_existing=False, delete_on_failure=True, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Download an object to the specified destination path.\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param object_name: Object name\n    :type  object_name: ``str``\n\n    :param destination_path: Full path to a file or a directory where the\n                                incoming file will be saved.\n    :type destination_path: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param overwrite_existing: True to overwrite an existing file,\n                                defaults to False.\n    :type overwrite_existing: ``bool``\n\n    :param delete_on_failure: True to delete a partially downloaded file if\n                                the download was not successful (hash\n                                mismatch / file size).\n    :type delete_on_failure: ``bool``\n\n    :param libcloud_kwargs: Extra arguments for the driver's download_object method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: True if an object has been successfully downloaded, False\n                otherwise.\n    :rtype: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.download_object MyFolder me.jpg /tmp/me.jpg profile1\n\n    "
    conn = _get_driver(profile=profile)
    obj = conn.get_object(container_name, object_name)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    return conn.download_object(obj, destination_path, overwrite_existing, delete_on_failure, **libcloud_kwargs)

def upload_object(file_path, container_name, object_name, profile, extra=None, verify_hash=True, headers=None, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Upload an object currently located on a disk.\n\n    :param file_path: Path to the object on disk.\n    :type file_path: ``str``\n\n    :param container_name: Destination container.\n    :type container_name: ``str``\n\n    :param object_name: Object name.\n    :type object_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param verify_hash: Verify hash\n    :type verify_hash: ``bool``\n\n    :param extra: Extra attributes (driver specific). (optional)\n    :type extra: ``dict``\n\n    :param headers: (optional) Additional request headers,\n        such as CORS headers. For example:\n        headers = {'Access-Control-Allow-Origin': 'http://mozilla.com'}\n    :type headers: ``dict``\n\n    :param libcloud_kwargs: Extra arguments for the driver's upload_object method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: The object name in the cloud\n    :rtype: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.upload_object /file/to/me.jpg MyFolder me.jpg profile1\n\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    container = conn.get_container(container_name)
    obj = conn.upload_object(file_path, container, object_name, extra, verify_hash, headers, **libcloud_kwargs)
    return obj.name

def delete_object(container_name, object_name, profile, **libcloud_kwargs):
    if False:
        print('Hello World!')
    "\n    Delete an object in the cloud\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param object_name: Object name\n    :type  object_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's delete_object method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: True if an object has been successfully deleted, False\n                otherwise.\n    :rtype: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.delete_object MyFolder me.jpg profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    obj = conn.get_object(container_name, object_name, **libcloud_kwargs)
    return conn.delete_object(obj)

def delete_container(container_name, profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    Delete an object container in the cloud\n\n    :param container_name: Container name\n    :type  container_name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's delete_container method\n    :type  libcloud_kwargs: ``dict``\n\n    :return: True if an object container has been successfully deleted, False\n                otherwise.\n    :rtype: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.delete_container MyFolder profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    container = conn.get_container(container_name)
    return conn.delete_container(container, **libcloud_kwargs)

def extra(method, profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Call an extended method on the driver\n\n    :param method: Driver's method name\n    :type  method: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's delete_container method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_storage.extra ex_get_permissions google container_name=my_container object_name=me.jpg --out=yaml\n    "
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    conn = _get_driver(profile=profile)
    connection_method = getattr(conn, method)
    return connection_method(**libcloud_kwargs)