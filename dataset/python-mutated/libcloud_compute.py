"""
Apache Libcloud Compute Management
==================================

Connection module for Apache Libcloud Compute management for a full list
of supported clouds, see http://libcloud.readthedocs.io/en/latest/compute/supported_providers.html

Clouds include Amazon EC2, Azure, Google GCE, VMware, OpenStack Nova

.. versionadded:: 2018.3.0

:configuration:
    This module uses a configuration profile for one or multiple cloud providers

    .. code-block:: yaml

        libcloud_compute:
            profile_test1:
              driver: google
              key: service-account@googlecloud.net
              secret: /path/to.key.json
            profile_test2:
              driver: arm
              key: 12345
              secret: mysecret

:depends: apache-libcloud
"""
import logging
import os.path
import salt.utils.args
import salt.utils.compat
from salt.utils.versions import Version
log = logging.getLogger(__name__)
REQUIRED_LIBCLOUD_VERSION = '2.0.0'
try:
    import libcloud
    from libcloud.compute.base import Node
    from libcloud.compute.providers import get_driver
    if hasattr(libcloud, '__version__') and Version(libcloud.__version__) < Version(REQUIRED_LIBCLOUD_VERSION):
        raise ImportError()
    logging.getLogger('libcloud').setLevel(logging.CRITICAL)
    HAS_LIBCLOUD = True
except ImportError:
    HAS_LIBCLOUD = False

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if libcloud libraries exist.\n    '
    if not HAS_LIBCLOUD:
        return (False, 'A apache-libcloud library with version at least {} was not found'.format(REQUIRED_LIBCLOUD_VERSION))
    return True

def _get_driver(profile):
    if False:
        while True:
            i = 10
    config = __salt__['config.option']('libcloud_compute')[profile]
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

def list_nodes(profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of nodes\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_nodes method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_nodes profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    nodes = conn.list_nodes(**libcloud_kwargs)
    ret = []
    for node in nodes:
        ret.append(_simple_node(node))
    return ret

def list_sizes(profile, location_id=None, **libcloud_kwargs):
    if False:
        return 10
    "\n    Return a list of node sizes\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param location_id: The location key, from list_locations\n    :type  location_id: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_sizes method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_sizes profile1\n        salt myminion libcloud_compute.list_sizes profile1 us-east1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    if location_id is not None:
        locations = [loc for loc in conn.list_locations() if loc.id == location_id]
        if not locations:
            raise ValueError('Location not found')
        else:
            sizes = conn.list_sizes(location=locations[0], **libcloud_kwargs)
    else:
        sizes = conn.list_sizes(**libcloud_kwargs)
    ret = []
    for size in sizes:
        ret.append(_simple_size(size))
    return ret

def list_locations(profile, **libcloud_kwargs):
    if False:
        print('Hello World!')
    "\n    Return a list of locations for this cloud\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_locations method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_locations profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    locations = conn.list_locations(**libcloud_kwargs)
    ret = []
    for loc in locations:
        ret.append(_simple_location(loc))
    return ret

def reboot_node(node_id, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Reboot a node in the cloud\n\n    :param node_id: Unique ID of the node to reboot\n    :type  node_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's reboot_node method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.reboot_node as-2346 profile1\n    "
    conn = _get_driver(profile=profile)
    node = _get_by_id(conn.list_nodes(**libcloud_kwargs), node_id)
    return conn.reboot_node(node, **libcloud_kwargs)

def destroy_node(node_id, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Destroy a node in the cloud\n\n    :param node_id: Unique ID of the node to destroy\n    :type  node_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's destroy_node method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.destry_node as-2346 profile1\n    "
    conn = _get_driver(profile=profile)
    node = _get_by_id(conn.list_nodes(**libcloud_kwargs), node_id)
    return conn.destroy_node(node, **libcloud_kwargs)

def list_volumes(profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    Return a list of storage volumes for this cloud\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_volumes method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_volumes profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volumes = conn.list_volumes(**libcloud_kwargs)
    ret = []
    for volume in volumes:
        ret.append(_simple_volume(volume))
    return ret

def list_volume_snapshots(volume_id, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of storage volumes snapshots for this cloud\n\n    :param volume_id: The volume identifier\n    :type  volume_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_volume_snapshots method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_volume_snapshots vol1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volume = _get_by_id(conn.list_volumes(), volume_id)
    snapshots = conn.list_volume_snapshots(volume, **libcloud_kwargs)
    ret = []
    for snapshot in snapshots:
        ret.append(_simple_volume_snapshot(snapshot))
    return ret

def create_volume(size, name, profile, location_id=None, **libcloud_kwargs):
    if False:
        return 10
    "\n    Create a storage volume\n\n    :param size: Size of volume in gigabytes (required)\n    :type size: ``int``\n\n    :param name: Name of the volume to be created\n    :type name: ``str``\n\n    :param location_id: Which data center to create a volume in. If\n                            empty, undefined behavior will be selected.\n                            (optional)\n    :type location_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_volumes method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.create_volume 1000 vol1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    if location_id is not None:
        location = _get_by_id(conn.list_locations(), location_id)
    else:
        location = None
    volume = conn.create_volume(size, name, location, snapshot=None, **libcloud_kwargs)
    return _simple_volume(volume)

def create_volume_snapshot(volume_id, profile, name=None, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Create a storage volume snapshot\n\n    :param volume_id:  Volume ID from which to create the new\n                        snapshot.\n    :type  volume_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param name: Name of the snapshot to be created (optional)\n    :type name: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's create_volume_snapshot method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.create_volume_snapshot vol1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volume = _get_by_id(conn.list_volumes(), volume_id)
    snapshot = conn.create_volume_snapshot(volume, name=name, **libcloud_kwargs)
    return _simple_volume_snapshot(snapshot)

def attach_volume(node_id, volume_id, profile, device=None, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Attaches volume to node.\n\n    :param node_id:  Node ID to target\n    :type  node_id: ``str``\n\n    :param volume_id:  Volume ID from which to attach\n    :type  volume_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param device: Where the device is exposed, e.g. '/dev/sdb'\n    :type device: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's attach_volume method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.detach_volume vol1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volume = _get_by_id(conn.list_volumes(), volume_id)
    node = _get_by_id(conn.list_nodes(), node_id)
    return conn.attach_volume(node, volume, device=device, **libcloud_kwargs)

def detach_volume(volume_id, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Detaches a volume from a node.\n\n    :param volume_id:  Volume ID from which to detach\n    :type  volume_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's detach_volume method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.detach_volume vol1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volume = _get_by_id(conn.list_volumes(), volume_id)
    return conn.detach_volume(volume, **libcloud_kwargs)

def destroy_volume(volume_id, profile, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Destroy a volume.\n\n    :param volume_id:  Volume ID from which to destroy\n    :type  volume_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's destroy_volume method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.destroy_volume vol1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volume = _get_by_id(conn.list_volumes(), volume_id)
    return conn.destroy_volume(volume, **libcloud_kwargs)

def destroy_volume_snapshot(volume_id, snapshot_id, profile, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Destroy a volume snapshot.\n\n    :param volume_id:  Volume ID from which the snapshot belongs\n    :type  volume_id: ``str``\n\n    :param snapshot_id:  Volume Snapshot ID from which to destroy\n    :type  snapshot_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's destroy_volume_snapshot method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.destroy_volume_snapshot snap1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    volume = _get_by_id(conn.list_volumes(), volume_id)
    snapshot = _get_by_id(conn.list_volume_snapshots(volume), snapshot_id)
    return conn.destroy_volume_snapshot(snapshot, **libcloud_kwargs)

def list_images(profile, location_id=None, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of images for this cloud\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param location_id: The location key, from list_locations\n    :type  location_id: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_images method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_images profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    if location_id is not None:
        location = _get_by_id(conn.list_locations(), location_id)
    else:
        location = None
    images = conn.list_images(location=location, **libcloud_kwargs)
    ret = []
    for image in images:
        ret.append(_simple_image(image))
    return ret

def create_image(node_id, name, profile, description=None, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Create an image from a node\n\n    :param node_id: Node to run the task on.\n    :type node_id: ``str``\n\n    :param name: name for new image.\n    :type name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param description: description for new image.\n    :type description: ``description``\n\n    :param libcloud_kwargs: Extra arguments for the driver's create_image method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.create_image server1 my_image profile1\n        salt myminion libcloud_compute.create_image server1 my_image profile1 description='test image'\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    node = _get_by_id(conn.list_nodes(), node_id)
    return _simple_image(conn.create_image(node, name, description=description, **libcloud_kwargs))

def delete_image(image_id, profile, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete an image of a node\n\n    :param image_id: Image to delete\n    :type image_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's delete_image method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.delete_image image1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    image = _get_by_id(conn.list_images(), image_id)
    return conn.delete_image(image, **libcloud_kwargs)

def get_image(image_id, profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    Get an image of a node\n\n    :param image_id: Image to fetch\n    :type image_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's delete_image method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.get_image image1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    image = conn.get_image(image_id, **libcloud_kwargs)
    return _simple_image(image)

def copy_image(source_region, image_id, name, profile, description=None, **libcloud_kwargs):
    if False:
        return 10
    "\n    Copies an image from a source region to the current region.\n\n    :param source_region: Region to copy the node from.\n    :type source_region: ``str``\n\n    :param image_id: Image to copy.\n    :type image_id: ``str``\n\n    :param name: name for new image.\n    :type name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param description: description for new image.\n    :type name: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's copy_image method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.copy_image us-east1 image1 'new image' profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    image = conn.get_image(image_id, **libcloud_kwargs)
    new_image = conn.copy_image(source_region, image, name, description=description, **libcloud_kwargs)
    return _simple_image(new_image)

def list_key_pairs(profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List all the available key pair objects.\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's list_key_pairs method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.list_key_pairs profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    keys = conn.list_key_pairs(**libcloud_kwargs)
    ret = []
    for key in keys:
        ret.append(_simple_key_pair(key))
    return ret

def get_key_pair(name, profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    Get a single key pair by name\n\n    :param name: Name of the key pair to retrieve.\n    :type name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's get_key_pair method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.get_key_pair pair1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    return _simple_key_pair(conn.get_key_pair(name, **libcloud_kwargs))

def create_key_pair(name, profile, **libcloud_kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a single key pair by name\n\n    :param name: Name of the key pair to create.\n    :type name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's create_key_pair method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.create_key_pair pair1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    return _simple_key_pair(conn.create_key_pair(name, **libcloud_kwargs))

def import_key_pair(name, key, profile, key_type=None, **libcloud_kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Import a new public key from string or a file path\n\n    :param name: Key pair name.\n    :type name: ``str``\n\n    :param key: Public key material, the string or a path to a file\n    :type  key: ``str`` or path ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param key_type: The key pair type, either `FILE` or `STRING`. Will detect if not provided\n        and assume that if the string is a path to an existing path it is a FILE, else STRING.\n    :type  key_type: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's import_key_pair_from_xxx method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.import_key_pair pair1 key_value_data123 profile1\n        salt myminion libcloud_compute.import_key_pair pair1 /path/to/key profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    if os.path.exists(key) or key_type == 'FILE':
        return _simple_key_pair(conn.import_key_pair_from_file(name, key, **libcloud_kwargs))
    else:
        return _simple_key_pair(conn.import_key_pair_from_string(name, key, **libcloud_kwargs))

def delete_key_pair(name, profile, **libcloud_kwargs):
    if False:
        return 10
    "\n    Delete a key pair\n\n    :param name: Key pair name.\n    :type  name: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's import_key_pair_from_xxx method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.delete_key_pair pair1 profile1\n    "
    conn = _get_driver(profile=profile)
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    key = conn.get_key_pair(name)
    return conn.delete_key_pair(key, **libcloud_kwargs)

def extra(method, profile, **libcloud_kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Call an extended method on the driver\n\n    :param method: Driver's method name\n    :type  method: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_compute.extra ex_get_permissions google container_name=my_container object_name=me.jpg --out=yaml\n    "
    libcloud_kwargs = salt.utils.args.clean_kwargs(**libcloud_kwargs)
    conn = _get_driver(profile=profile)
    connection_method = getattr(conn, method)
    return connection_method(**libcloud_kwargs)

def _get_by_id(collection, id):
    if False:
        print('Hello World!')
    '\n    Get item from a list by the id field\n    '
    matches = [item for item in collection if item.id == id]
    if not matches:
        raise ValueError('Could not find a matching item')
    elif len(matches) > 1:
        raise ValueError('The id matched {} items, not 1'.format(len(matches)))
    return matches[0]

def _simple_volume(volume):
    if False:
        while True:
            i = 10
    return {'id': volume.id, 'name': volume.name, 'size': volume.size, 'state': volume.state, 'extra': volume.extra}

def _simple_location(location):
    if False:
        print('Hello World!')
    return {'id': location.id, 'name': location.name, 'country': location.country}

def _simple_size(size):
    if False:
        i = 10
        return i + 15
    return {'id': size.id, 'name': size.name, 'ram': size.ram, 'disk': size.disk, 'bandwidth': size.bandwidth, 'price': size.price, 'extra': size.extra}

def _simple_node(node):
    if False:
        for i in range(10):
            print('nop')
    return {'id': node.id, 'name': node.name, 'state': str(node.state), 'public_ips': node.public_ips, 'private_ips': node.private_ips, 'size': _simple_size(node.size) if node.size else {}, 'extra': node.extra}

def _simple_volume_snapshot(snapshot):
    if False:
        print('Hello World!')
    return {'id': snapshot.id, 'name': snapshot.name if hasattr(snapshot, 'name') else snapshot.id, 'size': snapshot.size, 'extra': snapshot.extra, 'created': snapshot.created, 'state': snapshot.state}

def _simple_image(image):
    if False:
        i = 10
        return i + 15
    return {'id': image.id, 'name': image.name, 'extra': image.extra}

def _simple_key_pair(key):
    if False:
        return 10
    return {'name': key.name, 'fingerprint': key.fingerprint, 'public_key': key.public_key, 'private_key': key.private_key, 'extra': key.extra}