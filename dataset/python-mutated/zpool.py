"""
Module for running ZFS zpool command

:codeauthor:    Nitin Madhok <nmadhok@g.clemson.edu>, Jorge Schrauwen <sjorge@blackdot.be>
:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:depends:       salt.utils.zfs
:platform:      illumos,freebsd,linux

.. versionchanged:: 2018.3.1
  Big refactor to remove duplicate code, better type conversions and improved
  consistency in output.

"""
import logging
import os
import salt.utils.decorators
import salt.utils.decorators.path
import salt.utils.path
from salt.utils.odict import OrderedDict
log = logging.getLogger(__name__)
__virtualname__ = 'zpool'
__func_alias__ = {'import_': 'import', 'list_': 'list'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load when the platform has zfs support\n    '
    if __grains__.get('zfs_support'):
        return __virtualname__
    else:
        return (False, 'The zpool module cannot be loaded: zfs not supported')

def _clean_vdev_config(config):
    if False:
        print('Hello World!')
    "\n    Return a simple vdev tree from zpool.status' config section\n    "
    cln_config = OrderedDict()
    for (label, sub_config) in config.items():
        if label not in ['state', 'read', 'write', 'cksum']:
            sub_config = _clean_vdev_config(sub_config)
            if sub_config and isinstance(cln_config, list):
                cln_config.append(OrderedDict([(label, sub_config)]))
            elif sub_config and isinstance(cln_config, OrderedDict):
                cln_config[label] = sub_config
            elif isinstance(cln_config, list):
                cln_config.append(label)
            elif isinstance(cln_config, OrderedDict):
                new_config = []
                for (old_label, old_config) in cln_config.items():
                    new_config.append(OrderedDict([(old_label, old_config)]))
                new_config.append(label)
                cln_config = new_config
            else:
                cln_config = [label]
    return cln_config

def healthy():
    if False:
        i = 10
        return i + 15
    "\n    Check if all zpools are healthy\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.healthy\n\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command']('status', flags=['-x']), python_shell=False)
    return res['stdout'] == 'all pools are healthy'

def status(zpool=None):
    if False:
        while True:
            i = 10
    "\n    Return the status of the named zpool\n\n    zpool : string\n        optional name of storage pool\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.status myzpool\n\n    "
    ret = OrderedDict()
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command']('status', target=zpool), python_shell=False)
    if res['retcode'] != 0:
        return __utils__['zfs.parse_command_result'](res)
    current_pool = None
    current_prop = None
    for zpd in res['stdout'].splitlines():
        if zpd.strip() == '':
            continue
        if ':' in zpd and zpd[0] != '\t':
            prop = zpd.split(':')[0].strip()
            value = ':'.join(zpd.split(':')[1:]).strip()
            if prop == 'pool' and current_pool != value:
                current_pool = value
                ret[current_pool] = OrderedDict()
            if prop != 'pool':
                ret[current_pool][prop] = value
            current_prop = prop
        else:
            ret[current_pool][current_prop] = '{}\n{}'.format(ret[current_pool][current_prop], zpd)
    for pool in ret:
        if 'config' not in ret[pool]:
            continue
        header = None
        root_vdev = None
        vdev = None
        dev = None
        rdev = None
        config = ret[pool]['config']
        config_data = OrderedDict()
        for line in config.splitlines():
            if not header:
                header = line.strip().lower()
                header = [x for x in header.split(' ') if x not in ['']]
                continue
            if line[0] == '\t':
                line = line[1:]
            stat_data = OrderedDict(list(zip(header, [x for x in line.strip().split(' ') if x not in ['']])))
            stat_data = __utils__['zfs.from_auto_dict'](stat_data)
            if line.startswith(' ' * 6):
                rdev = stat_data['name']
                config_data[root_vdev][vdev][dev][rdev] = stat_data
            elif line.startswith(' ' * 4):
                rdev = None
                dev = stat_data['name']
                config_data[root_vdev][vdev][dev] = stat_data
            elif line.startswith(' ' * 2):
                rdev = dev = None
                vdev = stat_data['name']
                config_data[root_vdev][vdev] = stat_data
            else:
                rdev = dev = vdev = None
                root_vdev = stat_data['name']
                config_data[root_vdev] = stat_data
            del stat_data['name']
        ret[pool]['config'] = config_data
    return ret

def iostat(zpool=None, sample_time=5, parsable=True):
    if False:
        while True:
            i = 10
    "\n    Display I/O statistics for the given pools\n\n    zpool : string\n        optional name of storage pool\n\n    sample_time : int\n        seconds to capture data before output\n        default a sample of 5 seconds is used\n    parsable : boolean\n        display data in pythonc values (True, False, Bytes,...)\n\n    .. versionadded:: 2016.3.0\n    .. versionchanged:: 2018.3.1\n\n        Added ```parsable``` parameter that defaults to True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.iostat myzpool\n\n    "
    ret = OrderedDict()
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='iostat', flags=['-v'], target=[zpool, sample_time, 2]), python_shell=False)
    if res['retcode'] != 0:
        return __utils__['zfs.parse_command_result'](res)
    header = ['name', 'capacity-alloc', 'capacity-free', 'operations-read', 'operations-write', 'bandwidth-read', 'bandwidth-write']
    root_vdev = None
    vdev = None
    dev = None
    current_data = OrderedDict()
    for line in res['stdout'].splitlines():
        if line.strip() == '' or line.strip().split()[-1] in ['write', 'bandwidth']:
            continue
        if line.startswith('-') and line.endswith('-'):
            ret.update(current_data)
            current_data = OrderedDict()
            continue
        io_data = OrderedDict(list(zip(header, [x for x in line.strip().split(' ') if x not in ['']])))
        if parsable:
            io_data = __utils__['zfs.from_auto_dict'](io_data)
        else:
            io_data = __utils__['zfs.to_auto_dict'](io_data)
        if line.startswith(' ' * 4):
            dev = io_data['name']
            current_data[root_vdev][vdev][dev] = io_data
        elif line.startswith(' ' * 2):
            dev = None
            vdev = io_data['name']
            current_data[root_vdev][vdev] = io_data
        else:
            dev = vdev = None
            root_vdev = io_data['name']
            current_data[root_vdev] = io_data
        del io_data['name']
    return ret

def list_(properties='size,alloc,free,cap,frag,health', zpool=None, parsable=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2015.5.0\n\n    Return information about (all) storage pools\n\n    zpool : string\n        optional name of storage pool\n\n    properties : string\n        comma-separated list of properties to list\n\n    parsable : boolean\n        display numbers in parsable (exact) values\n\n        .. versionadded:: 2018.3.0\n\n    .. note::\n\n        The ``name`` property will always be included, while the ``frag``\n        property will get removed if not available\n\n    zpool : string\n        optional zpool\n\n    .. note::\n\n        Multiple storage pool can be provided as a space separated list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.list\n        salt '*' zpool.list zpool=tank\n        salt '*' zpool.list 'size,free'\n        salt '*' zpool.list 'size,free' tank\n\n    "
    ret = OrderedDict()
    if not isinstance(properties, list):
        properties = properties.split(',')
    while 'name' in properties:
        properties.remove('name')
    properties.insert(0, 'name')
    if not __utils__['zfs.has_feature_flags']():
        while 'frag' in properties:
            properties.remove('frag')
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='list', flags=['-H'], opts={'-o': ','.join(properties)}, target=zpool), python_shell=False)
    if res['retcode'] != 0:
        return __utils__['zfs.parse_command_result'](res)
    for line in res['stdout'].splitlines():
        zpool_data = OrderedDict(list(zip(properties, line.strip().split('\t'))))
        if parsable:
            zpool_data = __utils__['zfs.from_auto_dict'](zpool_data)
        else:
            zpool_data = __utils__['zfs.to_auto_dict'](zpool_data)
        ret[zpool_data['name']] = zpool_data
        del ret[zpool_data['name']]['name']
    return ret

def get(zpool, prop=None, show_source=False, parsable=True):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2016.3.0\n\n    Retrieves the given list of properties\n\n    zpool : string\n        Name of storage pool\n\n    prop : string\n        Optional name of property to retrieve\n\n    show_source : boolean\n        Show source of property\n\n    parsable : boolean\n        Display numbers in parsable (exact) values\n\n        .. versionadded:: 2018.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.get myzpool\n\n    "
    ret = OrderedDict()
    value_properties = ['name', 'property', 'value', 'source']
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='get', flags=['-H'], property_name=prop if prop else 'all', target=zpool), python_shell=False)
    if res['retcode'] != 0:
        return __utils__['zfs.parse_command_result'](res)
    for line in res['stdout'].splitlines():
        prop_data = OrderedDict(list(zip(value_properties, [x for x in line.strip().split('\t') if x not in ['']])))
        del prop_data['name']
        if parsable:
            prop_data['value'] = __utils__['zfs.from_auto'](prop_data['property'], prop_data['value'])
        else:
            prop_data['value'] = __utils__['zfs.to_auto'](prop_data['property'], prop_data['value'])
        if show_source:
            ret[prop_data['property']] = prop_data
            del ret[prop_data['property']]['property']
        else:
            ret[prop_data['property']] = prop_data['value']
    return ret

def set(zpool, prop, value):
    if False:
        i = 10
        return i + 15
    "\n    Sets the given property on the specified pool\n\n    zpool : string\n        Name of storage pool\n\n    prop : string\n        Name of property to set\n\n    value : string\n        Value to set for the specified property\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.set myzpool readonly yes\n\n    "
    ret = OrderedDict()
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='set', property_name=prop, property_value=value, target=zpool), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'set')

def exists(zpool):
    if False:
        i = 10
        return i + 15
    "\n    Check if a ZFS storage pool is active\n\n    zpool : string\n        Name of storage pool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.exists myzpool\n\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='list', target=zpool), python_shell=False, ignore_retcode=True)
    return res['retcode'] == 0

def destroy(zpool, force=False):
    if False:
        i = 10
        return i + 15
    "\n    Destroys a storage pool\n\n    zpool : string\n        Name of storage pool\n\n    force : boolean\n        Force destroy of pool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.destroy myzpool\n\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='destroy', flags=['-f'] if force else None, target=zpool), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'destroyed')

def scrub(zpool, stop=False, pause=False):
    if False:
        print('Hello World!')
    "\n    Scrub a storage pool\n\n    zpool : string\n        Name of storage pool\n\n    stop : boolean\n        If ``True``, cancel ongoing scrub\n\n    pause : boolean\n        If ``True``, pause ongoing scrub\n\n        .. versionadded:: 2018.3.0\n\n        .. note::\n\n            Pause is only available on recent versions of ZFS.\n\n            If both ``pause`` and ``stop`` are ``True``, then ``stop`` will\n            win.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.scrub myzpool\n\n    "
    if stop:
        action = ['-s']
    elif pause:
        action = ['-p']
    else:
        action = None
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='scrub', flags=action, target=zpool), python_shell=False)
    if res['retcode'] != 0:
        return __utils__['zfs.parse_command_result'](res, 'scrubbing')
    ret = OrderedDict()
    if stop or pause:
        ret['scrubbing'] = False
    else:
        ret['scrubbing'] = True
    return ret

def create(zpool, *vdevs, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 2015.5.0\n\n    Create a simple zpool, a mirrored zpool, a zpool having nested VDEVs, a hybrid zpool with cache, spare and log drives or a zpool with RAIDZ-1, RAIDZ-2 or RAIDZ-3\n\n    zpool : string\n        Name of storage pool\n\n    vdevs : string\n        One or move devices\n\n    force : boolean\n        Forces use of vdevs, even if they appear in use or specify a\n        conflicting replication level.\n\n    mountpoint : string\n        Sets the mount point for the root dataset\n\n    altroot : string\n        Equivalent to "-o cachefile=none,altroot=root"\n\n    properties : dict\n        Additional pool properties\n\n    filesystem_properties : dict\n        Additional filesystem properties\n\n    createboot : boolean\n        create a boot partition\n\n        .. versionadded:: 2018.3.0\n\n        .. warning:\n          This is only available on illumos and Solaris\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' zpool.create myzpool /path/to/vdev1 [...] [force=True|False]\n        salt \'*\' zpool.create myzpool mirror /path/to/vdev1 /path/to/vdev2 [...] [force=True|False]\n        salt \'*\' zpool.create myzpool raidz1 /path/to/vdev1 /path/to/vdev2 raidz2 /path/to/vdev3 /path/to/vdev4 /path/to/vdev5 [...] [force=True|False]\n        salt \'*\' zpool.create myzpool mirror /path/to/vdev1 [...] mirror /path/to/vdev2 /path/to/vdev3 [...] [force=True|False]\n        salt \'*\' zpool.create myhybridzpool mirror /tmp/file1 [...] log mirror /path/to/vdev1 [...] cache /path/to/vdev2 [...] spare /path/to/vdev3 [...] [force=True|False]\n\n    .. note::\n\n        Zpool properties can be specified at the time of creation of the pool\n        by passing an additional argument called "properties" and specifying\n        the properties with their respective values in the form of a python\n        dictionary:\n\n        .. code-block:: text\n\n            properties="{\'property1\': \'value1\', \'property2\': \'value2\'}"\n\n        Filesystem properties can be specified at the time of creation of the\n        pool by passing an additional argument called "filesystem_properties"\n        and specifying the properties with their respective values in the form\n        of a python dictionary:\n\n        .. code-block:: text\n\n            filesystem_properties="{\'property1\': \'value1\', \'property2\': \'value2\'}"\n\n        Example:\n\n        .. code-block:: bash\n\n            salt \'*\' zpool.create myzpool /path/to/vdev1 [...] properties="{\'property1\': \'value1\', \'property2\': \'value2\'}"\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zpool.create myzpool /path/to/vdev1 [...] [force=True|False]\n        salt \'*\' zpool.create myzpool mirror /path/to/vdev1 /path/to/vdev2 [...] [force=True|False]\n        salt \'*\' zpool.create myzpool raidz1 /path/to/vdev1 /path/to/vdev2 raidz2 /path/to/vdev3 /path/to/vdev4 /path/to/vdev5 [...] [force=True|False]\n        salt \'*\' zpool.create myzpool mirror /path/to/vdev1 [...] mirror /path/to/vdev2 /path/to/vdev3 [...] [force=True|False]\n        salt \'*\' zpool.create myhybridzpool mirror /tmp/file1 [...] log mirror /path/to/vdev1 [...] cache /path/to/vdev2 [...] spare /path/to/vdev3 [...] [force=True|False]\n\n    '
    flags = []
    opts = {}
    target = []
    pool_properties = kwargs.get('properties', {})
    filesystem_properties = kwargs.get('filesystem_properties', {})
    if kwargs.get('force', False):
        flags.append('-f')
    if kwargs.get('createboot', False) or 'bootsize' in pool_properties:
        flags.append('-B')
    if kwargs.get('altroot', False):
        opts['-R'] = kwargs.get('altroot')
    if kwargs.get('mountpoint', False):
        opts['-m'] = kwargs.get('mountpoint')
    target.append(zpool)
    target.extend(vdevs)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='create', flags=flags, opts=opts, pool_properties=pool_properties, filesystem_properties=filesystem_properties, target=target), python_shell=False)
    ret = __utils__['zfs.parse_command_result'](res, 'created')
    if ret['created']:
        ret['vdevs'] = _clean_vdev_config(__salt__['zpool.status'](zpool=zpool)[zpool]['config'][zpool])
    return ret

def add(zpool, *vdevs, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add the specified vdev's to the given storage pool\n\n    zpool : string\n        Name of storage pool\n\n    vdevs : string\n        One or more devices\n\n    force : boolean\n        Forces use of device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.add myzpool /path/to/vdev1 /path/to/vdev2 [...]\n\n    "
    flags = []
    target = []
    if kwargs.get('force', False):
        flags.append('-f')
    target.append(zpool)
    target.extend(vdevs)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='add', flags=flags, target=target), python_shell=False)
    ret = __utils__['zfs.parse_command_result'](res, 'added')
    if ret['added']:
        ret['vdevs'] = _clean_vdev_config(__salt__['zpool.status'](zpool=zpool)[zpool]['config'][zpool])
    return ret

def attach(zpool, device, new_device, force=False):
    if False:
        i = 10
        return i + 15
    "\n    Attach specified device to zpool\n\n    zpool : string\n        Name of storage pool\n\n    device : string\n        Existing device name too\n\n    new_device : string\n        New device name (to be attached to ``device``)\n\n    force : boolean\n        Forces use of device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.attach myzpool /path/to/vdev1 /path/to/vdev2 [...]\n\n    "
    flags = []
    target = []
    if force:
        flags.append('-f')
    target.append(zpool)
    target.append(device)
    target.append(new_device)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='attach', flags=flags, target=target), python_shell=False)
    ret = __utils__['zfs.parse_command_result'](res, 'attached')
    if ret['attached']:
        ret['vdevs'] = _clean_vdev_config(__salt__['zpool.status'](zpool=zpool)[zpool]['config'][zpool])
    return ret

def detach(zpool, device):
    if False:
        while True:
            i = 10
    "\n    Detach specified device to zpool\n\n    zpool : string\n        Name of storage pool\n\n    device : string\n        Device to detach\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.detach myzpool /path/to/vdev1\n\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='detach', target=[zpool, device]), python_shell=False)
    ret = __utils__['zfs.parse_command_result'](res, 'detatched')
    if ret['detatched']:
        ret['vdevs'] = _clean_vdev_config(__salt__['zpool.status'](zpool=zpool)[zpool]['config'][zpool])
    return ret

def split(zpool, newzpool, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 2018.3.0\n\n    Splits devices off pool creating newpool.\n\n    .. note::\n\n        All vdevs in pool must be mirrors.  At the time of the split,\n        ``newzpool`` will be a replica of ``zpool``.\n\n        After splitting, do not forget to import the new pool!\n\n    zpool : string\n        Name of storage pool\n\n    newzpool : string\n        Name of new storage pool\n\n    mountpoint : string\n        Sets the mount point for the root dataset\n\n    altroot : string\n        Sets altroot for newzpool\n\n    properties : dict\n        Additional pool properties for newzpool\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' zpool.split datamirror databackup\n        salt \'*\' zpool.split datamirror databackup altroot=/backup\n\n    .. note::\n\n        Zpool properties can be specified at the time of creation of the pool\n        by passing an additional argument called "properties" and specifying\n        the properties with their respective values in the form of a python\n        dictionary:\n\n        .. code-block:: text\n\n            properties="{\'property1\': \'value1\', \'property2\': \'value2\'}"\n\n        Example:\n\n        .. code-block:: bash\n\n            salt \'*\' zpool.split datamirror databackup properties="{\'readonly\': \'on\'}"\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zpool.split datamirror databackup\n        salt \'*\' zpool.split datamirror databackup altroot=/backup\n\n    '
    opts = {}
    pool_properties = kwargs.get('properties', {})
    if kwargs.get('altroot', False):
        opts['-R'] = kwargs.get('altroot')
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='split', opts=opts, pool_properties=pool_properties, target=[zpool, newzpool]), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'split')

def replace(zpool, old_device, new_device=None, force=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Replaces ``old_device`` with ``new_device``\n\n    .. note::\n\n        This is equivalent to attaching ``new_device``,\n        waiting for it to resilver, and then detaching ``old_device``.\n\n        The size of ``new_device`` must be greater than or equal to the minimum\n        size of all the devices in a mirror or raidz configuration.\n\n    zpool : string\n        Name of storage pool\n\n    old_device : string\n        Old device to replace\n\n    new_device : string\n        Optional new device\n\n    force : boolean\n        Forces use of new_device, even if its appears to be in use.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.replace myzpool /path/to/vdev1 /path/to/vdev2\n\n    "
    flags = []
    target = []
    if force:
        flags.append('-f')
    target.append(zpool)
    target.append(old_device)
    if new_device:
        target.append(new_device)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='replace', flags=flags, target=target), python_shell=False)
    ret = __utils__['zfs.parse_command_result'](res, 'replaced')
    if ret['replaced']:
        ret['vdevs'] = _clean_vdev_config(__salt__['zpool.status'](zpool=zpool)[zpool]['config'][zpool])
    return ret

@salt.utils.decorators.path.which('mkfile')
def create_file_vdev(size, *vdevs):
    if False:
        i = 10
        return i + 15
    "\n    Creates file based virtual devices for a zpool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.create_file_vdev 7G /path/to/vdev1 [/path/to/vdev2] [...]\n\n    .. note::\n\n        Depending on file size, the above command may take a while to return.\n\n    "
    ret = OrderedDict()
    err = OrderedDict()
    _mkfile_cmd = salt.utils.path.which('mkfile')
    for vdev in vdevs:
        if os.path.isfile(vdev):
            ret[vdev] = 'existed'
        else:
            res = __salt__['cmd.run_all']('{mkfile} {size} {vdev}'.format(mkfile=_mkfile_cmd, size=size, vdev=vdev), python_shell=False)
            if res['retcode'] != 0:
                if 'stderr' in res and ':' in res['stderr']:
                    ret[vdev] = 'failed'
                    err[vdev] = ':'.join(res['stderr'].strip().split(':')[1:])
            else:
                ret[vdev] = 'created'
    if err:
        ret['error'] = err
    return ret

def export(*pools, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2015.5.0\n\n    Export storage pools\n\n    pools : string\n        One or more storage pools to export\n\n    force : boolean\n        Force export of storage pools\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.export myzpool ... [force=True|False]\n        salt '*' zpool.export myzpool2 myzpool2 ... [force=True|False]\n\n    "
    flags = []
    targets = []
    if kwargs.get('force', False):
        flags.append('-f')
    targets = list(pools)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='export', flags=flags, target=targets), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'exported')

def import_(zpool=None, new_name=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2015.5.0\n\n    Import storage pools or list pools available for import\n\n    zpool : string\n        Optional name of storage pool\n\n    new_name : string\n        Optional new name for the storage pool\n\n    mntopts : string\n        Comma-separated list of mount options to use when mounting datasets\n        within the pool.\n\n    force : boolean\n        Forces import, even if the pool appears to be potentially active.\n\n    altroot : string\n        Equivalent to "-o cachefile=none,altroot=root"\n\n    dir : string\n        Searches for devices or files in dir, multiple dirs can be specified as\n        follows: ``dir="dir1,dir2"``\n\n    no_mount : boolean\n        Import the pool without mounting any file systems.\n\n    only_destroyed : boolean\n        Imports destroyed pools only. This also sets ``force=True``.\n\n    recovery : bool|str\n        false: do not try to recovery broken pools\n        true: try to recovery the pool by rolling back the latest transactions\n        test: check if a pool can be recovered, but don\'t import it\n        nolog: allow import without log device, recent transactions might be lost\n\n        .. note::\n            If feature flags are not support this forced to the default of \'false\'\n\n        .. warning::\n            When recovery is set to \'test\' the result will be have imported set to True if the pool\n            can be imported. The pool might also be imported if the pool was not broken to begin with.\n\n    properties : dict\n        Additional pool properties\n\n    .. note::\n\n        Zpool properties can be specified at the time of creation of the pool\n        by passing an additional argument called "properties" and specifying\n        the properties with their respective values in the form of a python\n        dictionary:\n\n        .. code-block:: text\n\n            properties="{\'property1\': \'value1\', \'property2\': \'value2\'}"\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zpool.import [force=True|False]\n        salt \'*\' zpool.import myzpool [mynewzpool] [force=True|False]\n        salt \'*\' zpool.import myzpool dir=\'/tmp\'\n\n    '
    flags = []
    opts = {}
    target = []
    pool_properties = kwargs.get('properties', {})
    if kwargs.get('force', False) or kwargs.get('only_destroyed', False):
        flags.append('-f')
    if kwargs.get('only_destroyed', False):
        flags.append('-D')
    if kwargs.get('no_mount', False):
        flags.append('-N')
    if kwargs.get('altroot', False):
        opts['-R'] = kwargs.get('altroot')
    if kwargs.get('mntopts', False):
        opts['-o'] = kwargs.get('mntopts')
    if kwargs.get('dir', False):
        opts['-d'] = kwargs.get('dir').split(',')
    if kwargs.get('recovery', False) and __utils__['zfs.has_feature_flags']():
        recovery = kwargs.get('recovery')
        if recovery in [True, 'test']:
            flags.append('-F')
        if recovery == 'test':
            flags.append('-n')
        if recovery == 'nolog':
            flags.append('-m')
    if zpool:
        target.append(zpool)
        target.append(new_name)
    else:
        flags.append('-a')
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='import', flags=flags, opts=opts, pool_properties=pool_properties, target=target), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'imported')

def online(zpool, *vdevs, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2015.5.0\n\n    Ensure that the specified devices are online\n\n    zpool : string\n        name of storage pool\n\n    vdevs : string\n        one or more devices\n\n    expand : boolean\n        Expand the device to use all available space.\n\n        .. note::\n\n            If the device is part of a mirror or raidz then all devices must be\n            expanded before the new space will become available to the pool.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.online myzpool /path/to/vdev1 [...]\n\n    "
    flags = []
    target = []
    if kwargs.get('expand', False):
        flags.append('-e')
    target.append(zpool)
    if vdevs:
        target.extend(vdevs)
    flags = []
    target = []
    if kwargs.get('expand', False):
        flags.append('-e')
    target.append(zpool)
    target.extend(vdevs)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='online', flags=flags, target=target), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'onlined')

def offline(zpool, *vdevs, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2015.5.0\n\n    Ensure that the specified devices are offline\n\n    .. warning::\n\n        By default, the ``OFFLINE`` state is persistent. The device remains\n        offline when the system is rebooted. To temporarily take a device\n        offline, use ``temporary=True``.\n\n    zpool : string\n        name of storage pool\n\n    vdevs : string\n        One or more devices\n\n    temporary : boolean\n        Enable temporarily offline\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.offline myzpool /path/to/vdev1 [...] [temporary=True|False]\n\n    "
    flags = []
    target = []
    if kwargs.get('temporary', False):
        flags.append('-t')
    target.append(zpool)
    target.extend(vdevs)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='offline', flags=flags, target=target), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'offlined')

def labelclear(device, force=False):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2018.3.0\n\n    Removes ZFS label information from the specified device\n\n    device : string\n        Device name; must not be part of an active pool configuration.\n\n    force : boolean\n        Treat exported or foreign devices as inactive\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.labelclear /path/to/dev\n\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='labelclear', flags=['-f'] if force else None, target=device), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'labelcleared')

def clear(zpool, device=None):
    if False:
        return 10
    "\n    Clears device errors in a pool.\n\n    .. warning::\n\n        The device must not be part of an active pool configuration.\n\n    zpool : string\n        name of storage pool\n    device : string\n        (optional) specific device to clear\n\n    .. versionadded:: 2018.3.1\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.clear mypool\n        salt '*' zpool.clear mypool /path/to/dev\n\n    "
    target = []
    target.append(zpool)
    target.append(device)
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='clear', target=target), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'cleared')

def reguid(zpool):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generates a new unique identifier for the pool\n\n    .. warning::\n        You must ensure that all devices in this pool are online and healthy\n        before performing this action.\n\n    zpool : string\n        name of storage pool\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.reguid myzpool\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='reguid', target=zpool), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'reguided')

def reopen(zpool):
    if False:
        return 10
    "\n    Reopen all the vdevs associated with the pool\n\n    zpool : string\n        name of storage pool\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.reopen myzpool\n\n    "
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='reopen', target=zpool), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'reopened')

def upgrade(zpool=None, version=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2016.3.0\n\n    Enables all supported features on the given pool\n\n    zpool : string\n        Optional storage pool, applies to all otherwize\n\n    version : int\n        Version to upgrade to, if unspecified upgrade to the highest possible\n\n    .. warning::\n        Once this is done, the pool will no longer be accessible on systems that do not\n        support feature flags. See zpool-features(5) for details on compatibility with\n        systems that support feature flags, but do not support all features enabled on the pool.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.upgrade myzpool\n\n    "
    flags = []
    opts = {}
    if version:
        opts['-V'] = version
    if not zpool:
        flags.append('-a')
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='upgrade', flags=flags, opts=opts, target=zpool), python_shell=False)
    return __utils__['zfs.parse_command_result'](res, 'upgraded')

def history(zpool=None, internal=False, verbose=False):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Displays the command history of the specified pools, or all pools if no\n    pool is specified\n\n    zpool : string\n        Optional storage pool\n\n    internal : boolean\n        Toggle display of internally logged ZFS events\n\n    verbose : boolean\n        Toggle display of the user name, the hostname, and the zone in which\n        the operation was performed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zpool.upgrade myzpool\n\n    "
    ret = OrderedDict()
    flags = []
    if verbose:
        flags.append('-l')
    if internal:
        flags.append('-i')
    res = __salt__['cmd.run_all'](__utils__['zfs.zpool_command'](command='history', flags=flags, target=zpool), python_shell=False)
    if res['retcode'] != 0:
        return __utils__['zfs.parse_command_result'](res)
    else:
        pool = 'unknown'
        for line in res['stdout'].splitlines():
            if line.startswith('History for'):
                pool = line[13:-2]
                ret[pool] = OrderedDict()
            else:
                if line == '':
                    continue
                log_timestamp = line[0:19]
                log_command = line[20:]
                ret[pool][log_timestamp] = log_command
    return ret