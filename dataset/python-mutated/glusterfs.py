"""
Manage a glusterfs pool
"""
import logging
import re
import sys
import xml.etree.ElementTree as ET
import salt.utils.cloud
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only load this module if the gluster command exists\n    '
    if salt.utils.path.which('gluster'):
        return True
    return (False, 'glusterfs server is not installed')

def _get_version():
    if False:
        return 10
    version = [3, 6]
    cmd = 'gluster --version'
    result = __salt__['cmd.run'](cmd).splitlines()
    for line in result:
        m = re.match('glusterfs ((?:\\d+\\.)+\\d+)', line)
        if m:
            version = m.group(1).split('.')
            version = [int(i) for i in version]
    return tuple(version)

def _gluster_ok(xml_data):
    if False:
        print('Hello World!')
    "\n    Extract boolean return value from Gluster's XML output.\n    "
    return int(xml_data.find('opRet').text) == 0

def _gluster_output_cleanup(result):
    if False:
        while True:
            i = 10
    '\n    Gluster versions prior to 6 have a bug that requires tricking\n    isatty. This adds "gluster> " to the output. Strip it off and\n    produce clean xml for ElementTree.\n    '
    ret = ''
    for line in result.splitlines():
        if line.startswith('gluster>'):
            ret += line[9:].strip()
        elif line.startswith('Welcome to gluster prompt'):
            pass
        else:
            ret += line.strip()
    return ret

def _gluster_xml(cmd):
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform a gluster --xml command and log result.\n    '
    if _get_version() < (3, 6):
        result = __salt__['cmd.run']('script -q -c "gluster --xml --mode=script"', stdin='{}\n\x04'.format(cmd))
    else:
        result = __salt__['cmd.run']('gluster --xml --mode=script', stdin='{}\n'.format(cmd))
    try:
        root = ET.fromstring(_gluster_output_cleanup(result))
    except ET.ParseError:
        raise CommandExecutionError('\n'.join(result.splitlines()[:-1]))
    if _gluster_ok(root):
        output = root.find('output')
        if output is not None:
            log.info('Gluster call "%s" succeeded: %s', cmd, root.find('output').text)
        else:
            log.info('Gluster call "%s" succeeded', cmd)
    else:
        log.error('Failed gluster call: %s: %s', cmd, root.find('opErrstr').text)
    return root

def _gluster(cmd):
    if False:
        i = 10
        return i + 15
    '\n    Perform a gluster command and return a boolean status.\n    '
    return _gluster_ok(_gluster_xml(cmd))

def _etree_to_dict(t):
    if False:
        i = 10
        return i + 15
    d = {}
    for child in t:
        d[child.tag] = _etree_to_dict(child)
    return d or t.text

def _iter(root, term):
    if False:
        return 10
    '\n    Checks for python2.6 or python2.7\n    '
    if sys.version_info < (2, 7):
        return root.getiterator(term)
    else:
        return root.iter(term)

def peer_status():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return peer status information\n\n    The return value is a dictionary with peer UUIDs as keys and dicts of peer\n    information as values. Hostnames are listed in one list. GlusterFS separates\n    one of the hostnames but the only reason for this seems to be which hostname\n    happens to be used first in peering.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.peer_status\n\n    GLUSTER direct CLI example (to show what salt is sending to gluster):\n\n        $ gluster peer status\n\n    GLUSTER CLI 3.4.4 return example (so we know what we are parsing):\n\n        Number of Peers: 2\n\n        Hostname: ftp2\n        Port: 24007\n        Uuid: cbcb256b-e66e-4ec7-a718-21082d396c24\n        State: Peer in Cluster (Connected)\n\n        Hostname: ftp3\n        Uuid: 5ea10457-6cb2-427b-a770-7897509625e9\n        State: Peer in Cluster (Connected)\n\n\n    "
    root = _gluster_xml('peer status')
    if not _gluster_ok(root):
        return None
    result = {}
    for peer in _iter(root, 'peer'):
        uuid = peer.find('uuid').text
        result[uuid] = {'hostnames': []}
        for item in peer:
            if item.tag == 'hostname':
                result[uuid]['hostnames'].append(item.text)
            elif item.tag == 'hostnames':
                for hostname in item:
                    if hostname.text not in result[uuid]['hostnames']:
                        result[uuid]['hostnames'].append(hostname.text)
            elif item.tag != 'uuid':
                result[uuid][item.tag] = item.text
    return result

def peer(name):
    if False:
        print('Hello World!')
    '\n    Add another node into the peer list.\n\n    name\n        The remote host to probe.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'one.gluster.*\' glusterfs.peer two\n\n    GLUSTER direct CLI example (to show what salt is sending to gluster):\n\n        $ gluster peer probe ftp2\n\n    GLUSTER CLI 3.4.4 return example (so we know what we are parsing):\n        #if the "peer" is the local host:\n        peer probe: success: on localhost not needed\n\n        #if the peer was just added:\n        peer probe: success\n\n        #if the peer was already part of the cluster:\n        peer probe: success: host ftp2 port 24007 already in peer list\n\n\n\n    '
    if salt.utils.cloud.check_name(name, 'a-zA-Z0-9._-'):
        raise SaltInvocationError('Invalid characters in peer name "{}"'.format(name))
    cmd = 'peer probe {}'.format(name)
    return _gluster(cmd)

def create_volume(name, bricks, stripe=False, replica=False, device_vg=False, transport='tcp', start=False, force=False, arbiter=False):
    if False:
        return 10
    '\n    Create a glusterfs volume\n\n    name\n        Name of the gluster volume\n\n    bricks\n        Bricks to create volume from, in <peer>:<brick path> format. For         multiple bricks use list format: \'["<peer1>:<brick1>",         "<peer2>:<brick2>"]\'\n\n    stripe\n        Stripe count, the number of bricks should be a multiple of the stripe         count for a distributed striped volume\n\n    replica\n        Replica count, the number of bricks should be a multiple of the         replica count for a distributed replicated volume\n\n    arbiter\n        If true, specifies volume should use arbiter brick(s).         Valid configuration limited to "replica 3 arbiter 1" per         Gluster documentation. Every third brick in the brick list         is used as an arbiter brick.\n\n        .. versionadded:: 2019.2.0\n\n    device_vg\n        If true, specifies volume should use block backend instead of regular         posix backend. Block device backend volume does not support multiple         bricks\n\n    transport\n        Transport protocol to use, can be \'tcp\', \'rdma\' or \'tcp,rdma\'\n\n    start\n        Start the volume after creation\n\n    force\n        Force volume creation, this works even if creating in root FS\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt host1 glusterfs.create newvolume host1:/brick\n\n        salt gluster1 glusterfs.create vol2 \'["gluster1:/export/vol2/brick",         "gluster2:/export/vol2/brick"]\' replica=2 start=True\n    '
    if isinstance(bricks, str):
        bricks = [bricks]
    if device_vg and len(bricks) > 1:
        raise SaltInvocationError('Block device backend volume does not ' + 'support multiple bricks')
    for brick in bricks:
        try:
            (peer_name, path) = brick.split(':')
            if not path.startswith('/'):
                raise SaltInvocationError('Brick paths must start with / in {}'.format(brick))
        except ValueError:
            raise SaltInvocationError('Brick syntax is <peer>:<path> got {}'.format(brick))
    if arbiter and replica != 3:
        raise SaltInvocationError('Arbiter configuration only valid ' + 'in replica 3 volume')
    cmd = 'volume create {} '.format(name)
    if stripe:
        cmd += 'stripe {} '.format(stripe)
    if replica:
        cmd += 'replica {} '.format(replica)
    if arbiter:
        cmd += 'arbiter 1 '
    if device_vg:
        cmd += 'device vg '
    if transport != 'tcp':
        cmd += 'transport {} '.format(transport)
    cmd += ' '.join(bricks)
    if force:
        cmd += ' force'
    if not _gluster(cmd):
        return False
    if start:
        return start_volume(name)
    return True

def list_volumes():
    if False:
        return 10
    "\n    List configured volumes\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.list_volumes\n    "
    root = _gluster_xml('volume list')
    if not _gluster_ok(root):
        return None
    results = [x.text for x in _iter(root, 'volume')]
    return results

def status(name):
    if False:
        i = 10
        return i + 15
    "\n    Check the status of a gluster volume.\n\n    name\n        Volume name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.status myvolume\n    "
    root = _gluster_xml('volume status {}'.format(name))
    if not _gluster_ok(root):
        return None
    ret = {'bricks': {}, 'nfs': {}, 'healers': {}}

    def etree_legacy_wrap(t):
        if False:
            for i in range(10):
                print('nop')
        ret = _etree_to_dict(t)
        ret['online'] = ret['status'] == '1'
        ret['host'] = ret['hostname']
        return ret
    hostref = {}
    for node in _iter(root, 'node'):
        peerid = node.find('peerid').text
        hostname = node.find('hostname').text
        if hostname not in ('NFS Server', 'Self-heal Daemon'):
            hostref[peerid] = hostname
    for node in _iter(root, 'node'):
        hostname = node.find('hostname').text
        if hostname not in ('NFS Server', 'Self-heal Daemon'):
            path = node.find('path').text
            ret['bricks']['{}:{}'.format(hostname, path)] = etree_legacy_wrap(node)
        elif hostname == 'NFS Server':
            peerid = node.find('peerid').text
            true_hostname = hostref[peerid]
            ret['nfs'][true_hostname] = etree_legacy_wrap(node)
        else:
            peerid = node.find('peerid').text
            true_hostname = hostref[peerid]
            ret['healers'][true_hostname] = etree_legacy_wrap(node)
    return ret

def info(name=None):
    if False:
        return 10
    "\n    .. versionadded:: 2015.8.4\n\n    Return gluster volume info.\n\n    name\n        Optional name to retrieve only information of one volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.info\n    "
    cmd = 'volume info'
    if name is not None:
        cmd += ' ' + name
    root = _gluster_xml(cmd)
    if not _gluster_ok(root):
        return None
    ret = {}
    for volume in _iter(root, 'volume'):
        name = volume.find('name').text
        ret[name] = _etree_to_dict(volume)
        bricks = {}
        for (i, brick) in enumerate(_iter(volume, 'brick'), start=1):
            brickkey = 'brick{}'.format(i)
            bricks[brickkey] = {'path': brick.text}
            for child in brick:
                if not child.tag == 'name':
                    bricks[brickkey].update({child.tag: child.text})
            for (k, v) in brick.items():
                bricks[brickkey][k] = v
        ret[name]['bricks'] = bricks
        options = {}
        for option in _iter(volume, 'option'):
            options[option.find('name').text] = option.find('value').text
        ret[name]['options'] = options
    return ret

def start_volume(name, force=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start a gluster volume\n\n    name\n        Volume name\n\n    force\n        Force the volume start even if the volume is started\n        .. versionadded:: 2015.8.4\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.start mycluster\n    "
    cmd = 'volume start {}'.format(name)
    if force:
        cmd = '{} force'.format(cmd)
    volinfo = info(name)
    if name not in volinfo:
        log.error('Cannot start non-existing volume %s', name)
        return False
    if not force and volinfo[name]['status'] == '1':
        log.info('Volume %s already started', name)
        return True
    return _gluster(cmd)

def stop_volume(name, force=False):
    if False:
        while True:
            i = 10
    "\n    Stop a gluster volume\n\n    name\n        Volume name\n\n    force\n        Force stop the volume\n\n        .. versionadded:: 2015.8.4\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.stop_volume mycluster\n    "
    volinfo = info()
    if name not in volinfo:
        log.error('Cannot stop non-existing volume %s', name)
        return False
    if int(volinfo[name]['status']) != 1:
        log.warning('Attempt to stop already stopped volume %s', name)
        return True
    cmd = 'volume stop {}'.format(name)
    if force:
        cmd += ' force'
    return _gluster(cmd)

def delete_volume(target, stop=True):
    if False:
        print('Hello World!')
    "\n    Deletes a gluster volume\n\n    target\n        Volume to delete\n\n    stop : True\n        If ``True``, stop volume before delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.delete_volume <volume>\n    "
    volinfo = info()
    if target not in volinfo:
        log.error('Cannot delete non-existing volume %s', target)
        return False
    running = volinfo[target]['status'] == '1'
    if not stop and running:
        log.error('Volume %s must be stopped before deletion', target)
        return False
    if running:
        if not stop_volume(target, force=True):
            return False
    cmd = 'volume delete {}'.format(target)
    return _gluster(cmd)

def add_volume_bricks(name, bricks):
    if False:
        while True:
            i = 10
    "\n    Add brick(s) to an existing volume\n\n    name\n        Volume name\n\n    bricks\n        List of bricks to add to the volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.add_volume_bricks <volume> <bricks>\n    "
    volinfo = info()
    if name not in volinfo:
        log.error('Volume %s does not exist, cannot add bricks', name)
        return False
    new_bricks = []
    cmd = 'volume add-brick {}'.format(name)
    if isinstance(bricks, str):
        bricks = [bricks]
    volume_bricks = [x['path'] for x in volinfo[name]['bricks'].values()]
    for brick in bricks:
        if brick in volume_bricks:
            log.debug('Brick %s already in volume %s...excluding from command', brick, name)
        else:
            new_bricks.append(brick)
    if new_bricks:
        for brick in new_bricks:
            cmd += ' {}'.format(brick)
        return _gluster(cmd)
    return True

def enable_quota_volume(name):
    if False:
        while True:
            i = 10
    "\n    Enable quota on a glusterfs volume.\n\n    name\n        Name of the gluster volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.enable_quota_volume <volume>\n    "
    cmd = 'volume quota {} enable'.format(name)
    if not _gluster(cmd):
        return False
    return True

def disable_quota_volume(name):
    if False:
        print('Hello World!')
    "\n    Disable quota on a glusterfs volume.\n\n    name\n        Name of the gluster volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.disable_quota_volume <volume>\n    "
    cmd = 'volume quota {} disable'.format(name)
    if not _gluster(cmd):
        return False
    return True

def set_quota_volume(name, path, size, enable_quota=False):
    if False:
        i = 10
        return i + 15
    '\n    Set quota to glusterfs volume.\n\n    name\n        Name of the gluster volume\n\n    path\n        Folder path for restriction in volume ("/")\n\n    size\n        Hard-limit size of the volume (MB/GB)\n\n    enable_quota\n        Enable quota before set up restriction\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' glusterfs.set_quota_volume <volume> <path> <size> enable_quota=True\n    '
    cmd = 'volume quota {}'.format(name)
    if path:
        cmd += ' limit-usage {}'.format(path)
    if size:
        cmd += ' {}'.format(size)
    if enable_quota:
        if not enable_quota_volume(name):
            pass
    if not _gluster(cmd):
        return False
    return True

def unset_quota_volume(name, path):
    if False:
        return 10
    "\n    Unset quota on glusterfs volume\n\n    name\n        Name of the gluster volume\n\n    path\n        Folder path for restriction in volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.unset_quota_volume <volume> <path>\n    "
    cmd = 'volume quota {}'.format(name)
    if path:
        cmd += ' remove {}'.format(path)
    if not _gluster(cmd):
        return False
    return True

def list_quota_volume(name):
    if False:
        i = 10
        return i + 15
    "\n    List quotas of glusterfs volume\n\n    name\n        Name of the gluster volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.list_quota_volume <volume>\n    "
    cmd = 'volume quota {}'.format(name)
    cmd += ' list'
    root = _gluster_xml(cmd)
    if not _gluster_ok(root):
        return None
    ret = {}
    for limit in _iter(root, 'limit'):
        path = limit.find('path').text
        ret[path] = _etree_to_dict(limit)
    return ret

def get_op_version(name):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Returns the glusterfs volume op-version\n\n    name\n        Name of the glusterfs volume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.get_op_version <volume>\n    "
    cmd = 'volume get {} cluster.op-version'.format(name)
    root = _gluster_xml(cmd)
    if not _gluster_ok(root):
        return (False, root.find('opErrstr').text)
    result = {}
    for op_version in _iter(root, 'volGetopts'):
        for item in op_version:
            if item.tag == 'Value':
                result = item.text
            elif item.tag == 'Opt':
                for child in item:
                    if child.tag == 'Value':
                        result = child.text
    return result

def get_max_op_version():
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Returns the glusterfs volume's max op-version value\n    Requires Glusterfs version > 3.9\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.get_max_op_version\n    "
    if _get_version() < (3, 10):
        return (False, 'Glusterfs version must be 3.10+.  Your version is {}.'.format(str('.'.join((str(i) for i in _get_version())))))
    cmd = 'volume get all cluster.max-op-version'
    root = _gluster_xml(cmd)
    if not _gluster_ok(root):
        return (False, root.find('opErrstr').text)
    result = {}
    for max_op_version in _iter(root, 'volGetopts'):
        for item in max_op_version:
            if item.tag == 'Value':
                result = item.text
            elif item.tag == 'Opt':
                for child in item:
                    if child.tag == 'Value':
                        result = child.text
    return result

def set_op_version(version):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Set the glusterfs volume op-version\n\n    version\n        Version to set the glusterfs volume op-version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.set_op_version <volume>\n    "
    cmd = 'volume set all cluster.op-version {}'.format(version)
    root = _gluster_xml(cmd)
    if not _gluster_ok(root):
        return (False, root.find('opErrstr').text)
    return root.find('output').text

def get_version():
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Returns the version of glusterfs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' glusterfs.get_version\n    "
    return '.'.join(_get_version())