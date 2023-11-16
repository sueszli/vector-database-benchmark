"""
Module to manage filesystem snapshots with snapper

.. versionadded:: 2016.11.0

:codeauthor:    Duncan Mac-Vicar P. <dmacvicar@suse.de>
:codeauthor:    Pablo Suárez Hernández <psuarezhernandez@suse.de>

:depends:       ``dbus`` Python module.
:depends:       ``snapper`` http://snapper.io, available in most distros
:maturity:      new
:platform:      Linux
"""
import difflib
import logging
import os
import subprocess
import time
import salt.utils.files
from salt.exceptions import CommandExecutionError
try:
    from pwd import getpwuid
    HAS_PWD = True
except ImportError:
    HAS_PWD = False
try:
    import dbus
    HAS_DBUS = True
except ImportError:
    HAS_DBUS = False
DBUS_STATUS_MAP = {1: 'created', 2: 'deleted', 4: 'type changed', 8: 'modified', 16: 'permission changed', 32: 'owner changed', 64: 'group changed', 128: 'extended attributes changed', 256: 'ACL info changed'}
SNAPPER_DBUS_OBJECT = 'org.opensuse.Snapper'
SNAPPER_DBUS_PATH = '/org/opensuse/Snapper'
SNAPPER_DBUS_INTERFACE = 'org.opensuse.Snapper'
log = logging.getLogger(__name__)
bus = None
system_bus_error = None
snapper = None
snapper_error = None
if HAS_DBUS:
    try:
        bus = dbus.SystemBus()
    except dbus.DBusException as exc:
        system_bus_error = exc
    else:
        if SNAPPER_DBUS_OBJECT in bus.list_activatable_names():
            try:
                snapper = dbus.Interface(bus.get_object(SNAPPER_DBUS_OBJECT, SNAPPER_DBUS_PATH), dbus_interface=SNAPPER_DBUS_INTERFACE)
            except (dbus.DBusException, ValueError) as exc:
                snapper_error = exc
        else:
            snapper_error = 'snapper is missing'

def __virtual__():
    if False:
        i = 10
        return i + 15
    error_msg = 'The snapper module cannot be loaded: {0}'
    if not HAS_DBUS:
        return (False, error_msg.format('missing python dbus module'))
    elif not snapper:
        return (False, error_msg.format(snapper_error))
    elif not bus:
        return (False, error_msg.format(system_bus_error))
    elif not HAS_PWD:
        return (False, error_msg.format('pwd module not available'))
    return 'snapper'

def _snapshot_to_data(snapshot):
    if False:
        while True:
            i = 10
    '\n    Returns snapshot data from a D-Bus response.\n\n    A snapshot D-Bus response is a dbus.Struct containing the\n    information related to a snapshot:\n\n    [id, type, pre_snapshot, timestamp, user, description,\n     cleanup_algorithm, userdata]\n\n    id: dbus.UInt32\n    type: dbus.UInt16\n    pre_snapshot: dbus.UInt32\n    timestamp: dbus.Int64\n    user: dbus.UInt32\n    description: dbus.String\n    cleaup_algorithm: dbus.String\n    userdata: dbus.Dictionary\n    '
    data = {}
    data['id'] = snapshot[0]
    data['type'] = ['single', 'pre', 'post'][snapshot[1]]
    if data['type'] == 'post':
        data['pre'] = snapshot[2]
    if snapshot[3] != -1:
        data['timestamp'] = snapshot[3]
    else:
        data['timestamp'] = int(time.time())
    data['user'] = getpwuid(snapshot[4])[0]
    data['description'] = snapshot[5]
    data['cleanup'] = snapshot[6]
    data['userdata'] = {}
    for (key, value) in snapshot[7].items():
        data['userdata'][key] = value
    return data

def _dbus_exception_to_reason(exc, args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a error message from a snapper DBusException\n    '
    error = exc.get_dbus_name()
    if error == 'error.unknown_config':
        return "Unknown configuration '{}'".format(args['config'])
    elif error == 'error.illegal_snapshot':
        return 'Invalid snapshot'
    else:
        return exc.get_dbus_name()

def list_snapshots(config='root'):
    if False:
        print('Hello World!')
    "\n    List available snapshots\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.list_snapshots config=myconfig\n    "
    try:
        snapshots = snapper.ListSnapshots(config)
        return [_snapshot_to_data(s) for s in snapshots]
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while listing snapshots: {}'.format(_dbus_exception_to_reason(exc, locals())))

def get_snapshot(number=0, config='root'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get detailed information about a given snapshot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.get_snapshot 1\n    "
    try:
        snapshot = snapper.GetSnapshot(config, int(number))
        return _snapshot_to_data(snapshot)
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while retrieving snapshot: {}'.format(_dbus_exception_to_reason(exc, locals())))

def list_configs():
    if False:
        while True:
            i = 10
    "\n    List all available configs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.list_configs\n    "
    try:
        configs = snapper.ListConfigs()
        return {config[0]: config[2] for config in configs}
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while listing configurations: {}'.format(_dbus_exception_to_reason(exc, locals())))

def _config_filter(value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return value

def set_config(name='root', **kwargs):
    if False:
        print('Hello World!')
    "\n    Set configuration values\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.set_config SYNC_ACL=True\n\n    Keys are case insensitive as they will be always uppercased to snapper\n    convention. The above example is equivalent to:\n\n    .. code-block:: bash\n\n        salt '*' snapper.set_config sync_acl=True\n    "
    try:
        data = {k.upper(): _config_filter(v) for (k, v) in kwargs.items() if not k.startswith('__')}
        snapper.SetConfig(name, data)
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while setting configuration {}: {}'.format(name, _dbus_exception_to_reason(exc, locals())))
    return True

def _get_last_snapshot(config='root'):
    if False:
        print('Hello World!')
    '\n    Returns the last existing created snapshot\n    '
    snapshot_list = sorted(list_snapshots(config), key=lambda x: x['id'])
    return snapshot_list[-1]

def status_to_string(dbus_status):
    if False:
        print('Hello World!')
    "\n    Converts a numeric dbus snapper status into a string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.status_to_string <dbus_status>\n    "
    status_tuple = (dbus_status & 1, dbus_status & 2, dbus_status & 4, dbus_status & 8, dbus_status & 16, dbus_status & 32, dbus_status & 64, dbus_status & 128, dbus_status & 256)
    return [DBUS_STATUS_MAP[status] for status in status_tuple if status]

def get_config(name='root'):
    if False:
        print('Hello World!')
    "\n    Retrieves all values from a given configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n      salt '*' snapper.get_config\n    "
    try:
        config = snapper.GetConfig(name)
        return config
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while retrieving configuration: {}'.format(_dbus_exception_to_reason(exc, locals())))

def create_config(name=None, subvolume=None, fstype=None, template=None, extra_opts=None):
    if False:
        print('Hello World!')
    '\n    Creates a new Snapper configuration\n\n    name\n        Name of the new Snapper configuration.\n    subvolume\n        Path to the related subvolume.\n    fstype\n        Filesystem type of the subvolume.\n    template\n        Configuration template to use. (Default: default)\n    extra_opts\n        Extra Snapper configuration opts dictionary. It will override the values provided\n        by the given template (if any).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n      salt \'*\' snapper.create_config name=myconfig subvolume=/foo/bar/ fstype=btrfs\n      salt \'*\' snapper.create_config name=myconfig subvolume=/foo/bar/ fstype=btrfs template="default"\n      salt \'*\' snapper.create_config name=myconfig subvolume=/foo/bar/ fstype=btrfs extra_opts=\'{"NUMBER_CLEANUP": False}\'\n    '

    def raise_arg_error(argname):
        if False:
            while True:
                i = 10
        raise CommandExecutionError('You must provide a "{}" for the new configuration'.format(argname))
    if not name:
        raise_arg_error('name')
    if not subvolume:
        raise_arg_error('subvolume')
    if not fstype:
        raise_arg_error('fstype')
    if not template:
        template = ''
    try:
        snapper.CreateConfig(name, subvolume, fstype, template)
        if extra_opts:
            set_config(name, **extra_opts)
        return get_config(name)
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while creating the new configuration: {}'.format(_dbus_exception_to_reason(exc, locals())))

def create_snapshot(config='root', snapshot_type='single', pre_number=None, description=None, cleanup_algorithm='number', userdata=None, **kwargs):
    if False:
        return 10
    "\n    Creates an snapshot\n\n    config\n        Configuration name.\n    snapshot_type\n        Specifies the type of the new snapshot. Possible values are\n        single, pre and post.\n    pre_number\n        For post snapshots the number of the pre snapshot must be\n        provided.\n    description\n        Description for the snapshot. If not given, the salt job will be used.\n    cleanup_algorithm\n        Set the cleanup algorithm for the snapshot.\n\n    number\n        Deletes old snapshots when a certain number of snapshots\n        is reached.\n    timeline\n        Deletes old snapshots but keeps a number of hourly,\n        daily, weekly, monthly and yearly snapshots.\n    empty-pre-post\n        Deletes pre/post snapshot pairs with empty diffs.\n    userdata\n        Set userdata for the snapshot (key-value pairs).\n\n    Returns the number of the created snapshot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.create_snapshot\n    "
    if not userdata:
        userdata = {}
    jid = kwargs.get('__pub_jid')
    if description is None and jid is not None:
        description = 'salt job {}'.format(jid)
    if jid is not None:
        userdata['salt_jid'] = jid
    new_nr = None
    try:
        if snapshot_type == 'single':
            new_nr = snapper.CreateSingleSnapshot(config, description, cleanup_algorithm, userdata)
        elif snapshot_type == 'pre':
            new_nr = snapper.CreatePreSnapshot(config, description, cleanup_algorithm, userdata)
        elif snapshot_type == 'post':
            if pre_number is None:
                raise CommandExecutionError("pre snapshot number 'pre_number' needs to bespecified for snapshots of the 'post' type")
            new_nr = snapper.CreatePostSnapshot(config, pre_number, description, cleanup_algorithm, userdata)
        else:
            raise CommandExecutionError("Invalid snapshot type '{}'".format(snapshot_type))
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while listing changed files: {}'.format(_dbus_exception_to_reason(exc, locals())))
    return new_nr

def delete_snapshot(snapshots_ids=None, config='root'):
    if False:
        i = 10
        return i + 15
    "\n    Deletes an snapshot\n\n    config\n        Configuration name. (Default: root)\n\n    snapshots_ids\n        List of the snapshots IDs to be deleted.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.delete_snapshot 54\n        salt '*' snapper.delete_snapshot config=root 54\n        salt '*' snapper.delete_snapshot config=root snapshots_ids=[54,55,56]\n    "
    if not snapshots_ids:
        raise CommandExecutionError('Error: No snapshot ID has been provided')
    try:
        current_snapshots_ids = [x['id'] for x in list_snapshots(config)]
        if not isinstance(snapshots_ids, list):
            snapshots_ids = [snapshots_ids]
        if not set(snapshots_ids).issubset(set(current_snapshots_ids)):
            raise CommandExecutionError("Error: Snapshots '{}' not found".format(', '.join([str(x) for x in set(snapshots_ids).difference(set(current_snapshots_ids))])))
        snapper.DeleteSnapshots(config, snapshots_ids)
        return {config: {'ids': snapshots_ids, 'status': 'deleted'}}
    except dbus.DBusException as exc:
        raise CommandExecutionError(_dbus_exception_to_reason(exc, locals()))

def modify_snapshot(snapshot_id=None, description=None, userdata=None, cleanup=None, config='root'):
    if False:
        print('Hello World!')
    '\n    Modify attributes of an existing snapshot.\n\n    config\n        Configuration name. (Default: root)\n\n    snapshot_id\n        ID of the snapshot to be modified.\n\n    cleanup\n        Change the cleanup method of the snapshot. (str)\n\n    description\n        Change the description of the snapshot. (str)\n\n    userdata\n        Change the userdata dictionary of the snapshot. (dict)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' snapper.modify_snapshot 54 description="my snapshot description"\n        salt \'*\' snapper.modify_snapshot 54 description="my snapshot description"\n        salt \'*\' snapper.modify_snapshot 54 userdata=\'{"foo": "bar"}\'\n        salt \'*\' snapper.modify_snapshot snapshot_id=54 cleanup="number"\n    '
    if not snapshot_id:
        raise CommandExecutionError('Error: No snapshot ID has been provided')
    snapshot = get_snapshot(config=config, number=snapshot_id)
    try:
        updated_opts = {'description': description if description is not None else snapshot['description'], 'cleanup': cleanup if cleanup is not None else snapshot['cleanup'], 'userdata': userdata if userdata is not None else snapshot['userdata']}
        snapper.SetSnapshot(config, snapshot_id, updated_opts['description'], updated_opts['cleanup'], updated_opts['userdata'])
        return get_snapshot(config=config, number=snapshot_id)
    except dbus.DBusException as exc:
        raise CommandExecutionError(_dbus_exception_to_reason(exc, locals()))

def _get_num_interval(config, num_pre, num_post):
    if False:
        return 10
    '\n    Returns numerical interval based on optionals num_pre, num_post values\n    '
    post = int(num_post) if num_post else 0
    pre = int(num_pre) if num_pre is not None else _get_last_snapshot(config)['id']
    return (pre, post)

def _is_text_file(filename):
    if False:
        print('Hello World!')
    '\n    Checks if a file is a text file\n    '
    type_of_file = subprocess.run(['file', '-bi', filename], check=False, stdout=subprocess.PIPE, universal_newlines=True).stdout
    return type_of_file.startswith('text')

def run(function, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Runs a function from an execution module creating pre and post snapshots\n    and associating the salt job id with those snapshots for easy undo and\n    cleanup.\n\n    function\n        Salt function to call.\n\n    config\n        Configuration name. (default: "root")\n\n    description\n        A description for the snapshots. (default: None)\n\n    userdata\n        Data to include in the snapshot metadata. (default: None)\n\n    cleanup_algorithm\n        Snapper cleanup algorithm. (default: "number")\n\n    `*args`\n        args for the function to call. (default: None)\n\n    `**kwargs`\n        kwargs for the function to call (default: None)\n\n    This  would run append text to /etc/motd using the file.append\n    module, and will create two snapshots, pre and post with the associated\n    metadata. The jid will be available as salt_jid in the userdata of the\n    snapshot.\n\n    You can immediately see the changes\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' snapper.run file.append args=\'["/etc/motd", "some text"]\'\n    '
    config = kwargs.pop('config', 'root')
    description = kwargs.pop('description', 'snapper.run[{}]'.format(function))
    cleanup_algorithm = kwargs.pop('cleanup_algorithm', 'number')
    userdata = kwargs.pop('userdata', {})
    func_kwargs = {k: v for (k, v) in kwargs.items() if not k.startswith('__')}
    kwargs = {k: v for (k, v) in kwargs.items() if k.startswith('__')}
    pre_nr = __salt__['snapper.create_snapshot'](config=config, snapshot_type='pre', description=description, cleanup_algorithm=cleanup_algorithm, userdata=userdata, **kwargs)
    if function not in __salt__:
        raise CommandExecutionError('function "{}" does not exist'.format(function))
    try:
        ret = __salt__[function](*args, **func_kwargs)
    except CommandExecutionError as exc:
        ret = '\n'.join([str(exc), __salt__[function].__doc__])
    __salt__['snapper.create_snapshot'](config=config, snapshot_type='post', pre_number=pre_nr, description=description, cleanup_algorithm=cleanup_algorithm, userdata=userdata, **kwargs)
    return ret

def status(config='root', num_pre=None, num_post=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns a comparison between two snapshots\n\n    config\n        Configuration name.\n\n    num_pre\n        first snapshot ID to compare. Default is last snapshot\n\n    num_post\n        last snapshot ID to compare. Default is 0 (current state)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.status\n        salt '*' snapper.status num_pre=19 num_post=20\n    "
    try:
        (pre, post) = _get_num_interval(config, num_pre, num_post)
        snapper.CreateComparison(config, int(pre), int(post))
        files = snapper.GetFiles(config, int(pre), int(post))
        status_ret = {}
        subvolume = list_configs()[config]['SUBVOLUME']
        for file in files:
            _filepath = file[0][len(subvolume):] if file[0].startswith(subvolume) else file[0]
            status_ret[os.path.normpath(subvolume + _filepath)] = {'status': status_to_string(file[1])}
        return status_ret
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while listing changed files: {}'.format(_dbus_exception_to_reason(exc, locals())))

def changed_files(config='root', num_pre=None, num_post=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the files changed between two snapshots\n\n    config\n        Configuration name.\n\n    num_pre\n        first snapshot ID to compare. Default is last snapshot\n\n    num_post\n        last snapshot ID to compare. Default is 0 (current state)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.changed_files\n        salt '*' snapper.changed_files num_pre=19 num_post=20\n    "
    return status(config, num_pre, num_post).keys()

def undo(config='root', files=None, num_pre=None, num_post=None):
    if False:
        while True:
            i = 10
    "\n    Undo all file changes that happened between num_pre and num_post, leaving\n    the files into the state of num_pre.\n\n    .. warning::\n        If one of the files has changes after num_post, they will be overwritten\n        The snapshots are used to determine the file list, but the current\n        version of the files will be overwritten by the versions in num_pre.\n\n        You to undo changes between num_pre and the current version of the\n        files use num_post=0.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.undo\n    "
    (pre, post) = _get_num_interval(config, num_pre, num_post)
    changes = status(config, pre, post)
    changed = set(changes.keys())
    requested = set(files or changed)
    if not requested.issubset(changed):
        raise CommandExecutionError('Given file list contains files that are not presentin the changed filelist: {}'.format(changed - requested))
    cmdret = __salt__['cmd.run']('snapper -c {} undochange {}..{} {}'.format(config, pre, post, ' '.join(requested)))
    try:
        components = cmdret.split(' ')
        ret = {}
        for comp in components:
            (key, val) = comp.split(':')
            ret[key] = val
        return ret
    except ValueError as exc:
        raise CommandExecutionError('Error while processing Snapper response: {}'.format(cmdret))

def _get_jid_snapshots(jid, config='root'):
    if False:
        i = 10
        return i + 15
    "\n    Returns pre/post snapshots made by a given Salt jid\n\n    Looks for 'salt_jid' entries into snapshots userdata which are created\n    when 'snapper.run' is executed.\n    "
    jid_snapshots = [x for x in list_snapshots(config) if x['userdata'].get('salt_jid') == jid]
    pre_snapshot = [x for x in jid_snapshots if x['type'] == 'pre']
    post_snapshot = [x for x in jid_snapshots if x['type'] == 'post']
    if not pre_snapshot or not post_snapshot:
        raise CommandExecutionError("Jid '{}' snapshots not found".format(jid))
    return (pre_snapshot[0]['id'], post_snapshot[0]['id'])

def undo_jid(jid, config='root'):
    if False:
        while True:
            i = 10
    "\n    Undo the changes applied by a salt job\n\n    jid\n        The job id to lookup\n\n    config\n        Configuration name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.undo_jid jid=20160607130930720112\n    "
    (pre_snapshot, post_snapshot) = _get_jid_snapshots(jid, config=config)
    return undo(config, num_pre=pre_snapshot, num_post=post_snapshot)

def diff(config='root', filename=None, num_pre=None, num_post=None):
    if False:
        while True:
            i = 10
    '\n    Returns the differences between two snapshots\n\n    config\n        Configuration name.\n\n    filename\n        if not provided the showing differences between snapshots for\n        all "text" files\n\n    num_pre\n        first snapshot ID to compare. Default is last snapshot\n\n    num_post\n        last snapshot ID to compare. Default is 0 (current state)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' snapper.diff\n        salt \'*\' snapper.diff filename=/var/log/snapper.log num_pre=19 num_post=20\n    '
    try:
        (pre, post) = _get_num_interval(config, num_pre, num_post)
        files = changed_files(config, pre, post)
        if filename:
            files = [filename] if filename in files else []
        subvolume = list_configs()[config]['SUBVOLUME']
        pre_mount = snapper.MountSnapshot(config, pre, False) if pre else subvolume
        post_mount = snapper.MountSnapshot(config, post, False) if post else subvolume
        files_diff = dict()
        for filepath in [filepath for filepath in files if not os.path.isdir(filepath)]:
            _filepath = filepath
            if filepath.startswith(subvolume):
                _filepath = filepath[len(subvolume):]
            pre_file = os.path.normpath(pre_mount + '/' + _filepath).replace('//', '/')
            post_file = os.path.normpath(post_mount + '/' + _filepath).replace('//', '/')
            if os.path.isfile(pre_file):
                pre_file_exists = True
                with salt.utils.files.fopen(pre_file) as rfh:
                    pre_file_content = [salt.utils.stringutils.to_unicode(_l) for _l in rfh.readlines()]
            else:
                pre_file_content = []
                pre_file_exists = False
            if os.path.isfile(post_file):
                post_file_exists = True
                with salt.utils.files.fopen(post_file) as rfh:
                    post_file_content = [salt.utils.stringutils.to_unicode(_l) for _l in rfh.readlines()]
            else:
                post_file_content = []
                post_file_exists = False
            if _is_text_file(pre_file) or _is_text_file(post_file):
                files_diff[filepath] = {'comment': 'text file changed', 'diff': ''.join(difflib.unified_diff(pre_file_content, post_file_content, fromfile=pre_file, tofile=post_file))}
                if pre_file_exists and (not post_file_exists):
                    files_diff[filepath]['comment'] = 'text file deleted'
                if not pre_file_exists and post_file_exists:
                    files_diff[filepath]['comment'] = 'text file created'
            elif not _is_text_file(pre_file) and (not _is_text_file(post_file)):
                files_diff[filepath] = {'comment': 'binary file changed'}
                if pre_file_exists:
                    files_diff[filepath]['old_sha256_digest'] = __salt__['hashutil.sha256_digest'](''.join(pre_file_content))
                if post_file_exists:
                    files_diff[filepath]['new_sha256_digest'] = __salt__['hashutil.sha256_digest'](''.join(post_file_content))
                if post_file_exists and (not pre_file_exists):
                    files_diff[filepath]['comment'] = 'binary file created'
                if pre_file_exists and (not post_file_exists):
                    files_diff[filepath]['comment'] = 'binary file deleted'
        if pre:
            snapper.UmountSnapshot(config, pre, False)
        if post:
            snapper.UmountSnapshot(config, post, False)
        return files_diff
    except dbus.DBusException as exc:
        raise CommandExecutionError('Error encountered while showing differences between snapshots: {}'.format(_dbus_exception_to_reason(exc, locals())))

def diff_jid(jid, config='root'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the changes applied by a `jid`\n\n    jid\n        The job id to lookup\n\n    config\n        Configuration name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.diff_jid jid=20160607130930720112\n    "
    (pre_snapshot, post_snapshot) = _get_jid_snapshots(jid, config=config)
    return diff(config, num_pre=pre_snapshot, num_post=post_snapshot)

def create_baseline(tag='baseline', config='root'):
    if False:
        i = 10
        return i + 15
    "\n    Creates a snapshot marked as baseline\n\n    tag\n        Tag name for the baseline\n\n    config\n        Configuration name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' snapper.create_baseline\n        salt '*' snapper.create_baseline my_custom_baseline\n    "
    return __salt__['snapper.create_snapshot'](config=config, snapshot_type='single', description='baseline snapshot', cleanup_algorithm='number', userdata={'baseline_tag': tag})