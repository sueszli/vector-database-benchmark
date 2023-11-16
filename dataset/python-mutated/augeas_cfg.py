"""
Manages configuration files via augeas

This module requires the ``augeas`` Python module.

.. _Augeas: http://augeas.net/

.. warning::

    Minimal installations of Debian and Ubuntu have been seen to have packaging
    bugs with python-augeas, causing the augeas module to fail to import. If
    the minion has the augeas module installed, but the functions in this
    execution module fail to run due to being unavailable, first restart the
    salt-minion service. If the problem persists past that, the following
    command can be run from the master to determine what is causing the import
    to fail:

    .. code-block:: bash

        salt minion-id cmd.run 'python -c "from augeas import Augeas"'

    For affected Debian/Ubuntu hosts, installing ``libpython2.7`` has been
    known to resolve the issue.
"""
import logging
import os
import re
import salt.utils.args
import salt.utils.data
import salt.utils.stringutils
from salt.exceptions import SaltInvocationError
HAS_AUGEAS = False
try:
    from augeas import Augeas as _Augeas
    HAS_AUGEAS = True
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'augeas'
METHOD_MAP = {'set': 'set', 'setm': 'setm', 'mv': 'move', 'move': 'move', 'ins': 'insert', 'insert': 'insert', 'rm': 'remove', 'remove': 'remove'}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only run this module if the augeas python module is installed\n    '
    if HAS_AUGEAS:
        return __virtualname__
    return (False, 'Cannot load augeas_cfg module: augeas python module not installed')

def _recurmatch(path, aug):
    if False:
        i = 10
        return i + 15
    '\n    Recursive generator providing the infrastructure for\n    augtools print behavior.\n\n    This function is based on test_augeas.py from\n    Harald Hoyer <harald@redhat.com>  in the python-augeas\n    repository\n    '
    if path:
        clean_path = path.rstrip('/*')
        yield (clean_path, aug.get(path))
        for i in aug.match(clean_path + '/*'):
            i = i.replace('!', '\\!')
            yield from _recurmatch(i, aug)

def _lstrip_word(word, prefix):
    if False:
        i = 10
        return i + 15
    '\n    Return a copy of the string after the specified prefix was removed\n    from the beginning of the string\n    '
    if str(word).startswith(prefix):
        return str(word)[len(prefix):]
    return word

def _check_load_paths(load_path):
    if False:
        print('Hello World!')
    '\n    Checks the validity of the load_path, returns a sanitized version\n    with invalid paths removed.\n    '
    if load_path is None or not isinstance(load_path, str):
        return None
    _paths = []
    for _path in load_path.split(':'):
        if os.path.isabs(_path) and os.path.isdir(_path):
            _paths.append(_path)
        else:
            log.info('Invalid augeas_cfg load_path entry: %s removed', _path)
    if not _paths:
        return None
    return ':'.join(_paths)

def execute(context=None, lens=None, commands=(), load_path=None):
    if False:
        i = 10
        return i + 15
    '\n    Execute Augeas commands\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' augeas.execute /files/etc/redis/redis.conf \\\n        commands=\'["set bind 0.0.0.0", "set maxmemory 1G"]\'\n\n    context\n        The Augeas context\n\n    lens\n        The Augeas lens to use\n\n    commands\n        The Augeas commands to execute\n\n    .. versionadded:: 2016.3.0\n\n    load_path\n        A colon-spearated list of directories that modules should be searched\n        in. This is in addition to the standard load path and the directories\n        in AUGEAS_LENS_LIB.\n    '
    ret = {'retval': False}
    arg_map = {'set': (1, 2), 'setm': (2, 3), 'move': (2,), 'insert': (3,), 'remove': (1,)}

    def make_path(path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return correct path\n        '
        if not context:
            return path
        if path.lstrip('/'):
            if path.startswith(context):
                return path
            path = path.lstrip('/')
            return os.path.join(context, path)
        else:
            return context
    load_path = _check_load_paths(load_path)
    flags = _Augeas.NO_MODL_AUTOLOAD if lens and context else _Augeas.NONE
    aug = _Augeas(flags=flags, loadpath=load_path)
    if lens and context:
        aug.add_transform(lens, re.sub('^/files', '', context))
        aug.load()
    for command in commands:
        try:
            (cmd, arg) = command.split(' ', 1)
            if cmd not in METHOD_MAP:
                ret['error'] = 'Command {} is not supported (yet)'.format(cmd)
                return ret
            method = METHOD_MAP[cmd]
            nargs = arg_map[method]
            parts = salt.utils.args.shlex_split(arg)
            if len(parts) not in nargs:
                err = '{} takes {} args: {}'.format(method, nargs, parts)
                raise ValueError(err)
            if method == 'set':
                path = make_path(parts[0])
                value = parts[1] if len(parts) == 2 else None
                args = {'path': path, 'value': value}
            elif method == 'setm':
                base = make_path(parts[0])
                sub = parts[1]
                value = parts[2] if len(parts) == 3 else None
                args = {'base': base, 'sub': sub, 'value': value}
            elif method == 'move':
                path = make_path(parts[0])
                dst = parts[1]
                args = {'src': path, 'dst': dst}
            elif method == 'insert':
                (label, where, path) = parts
                if where not in ('before', 'after'):
                    raise ValueError('Expected "before" or "after", not {}'.format(where))
                path = make_path(path)
                args = {'path': path, 'label': label, 'before': where == 'before'}
            elif method == 'remove':
                path = make_path(parts[0])
                args = {'path': path}
        except ValueError as err:
            log.error(err)
            if 'arg' not in locals():
                arg = command
            ret['error'] = 'Invalid formatted command, see debug log for details: {}'.format(arg)
            return ret
        args = salt.utils.data.decode(args, to_str=True)
        log.debug('%s: %s', method, args)
        func = getattr(aug, method)
        func(**args)
    try:
        aug.save()
        ret['retval'] = True
    except OSError as err:
        ret['error'] = str(err)
        if lens and (not lens.endswith('.lns')):
            ret['error'] += '\nLenses are normally configured as "name.lns". Did you mean "{}.lns"?'.format(lens)
    aug.close()
    return ret

def get(path, value='', load_path=None):
    if False:
        while True:
            i = 10
    "\n    Get a value for a specific augeas path\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' augeas.get /files/etc/hosts/1/ ipaddr\n\n    path\n        The path to get the value of\n\n    value\n        The optional value to get\n\n    .. versionadded:: 2016.3.0\n\n    load_path\n        A colon-spearated list of directories that modules should be searched\n        in. This is in addition to the standard load path and the directories\n        in AUGEAS_LENS_LIB.\n    "
    load_path = _check_load_paths(load_path)
    aug = _Augeas(loadpath=load_path)
    ret = {}
    path = path.rstrip('/')
    if value:
        path += '/{}'.format(value.strip('/'))
    try:
        _match = aug.match(path)
    except RuntimeError as err:
        return {'error': str(err)}
    if _match:
        ret[path] = aug.get(path)
    else:
        ret[path] = ''
    return ret

def setvalue(*args):
    if False:
        print('Hello World!')
    '\n    Set a value for a specific augeas path\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' augeas.setvalue /files/etc/hosts/1/canonical localhost\n\n    This will set the first entry in /etc/hosts to localhost\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' augeas.setvalue /files/etc/hosts/01/ipaddr 192.168.1.1 \\\n                                 /files/etc/hosts/01/canonical test\n\n    Adds a new host to /etc/hosts the ip address 192.168.1.1 and hostname test\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' augeas.setvalue prefix=/files/etc/sudoers/ \\\n                 "spec[user = \'%wheel\']/user" "%wheel" \\\n                 "spec[user = \'%wheel\']/host_group/host" \'ALL\' \\\n                 "spec[user = \'%wheel\']/host_group/command[1]" \'ALL\' \\\n                 "spec[user = \'%wheel\']/host_group/command[1]/tag" \'PASSWD\' \\\n                 "spec[user = \'%wheel\']/host_group/command[2]" \'/usr/bin/apt-get\' \\\n                 "spec[user = \'%wheel\']/host_group/command[2]/tag" NOPASSWD\n\n    Ensures that the following line is present in /etc/sudoers::\n\n        %wheel ALL = PASSWD : ALL , NOPASSWD : /usr/bin/apt-get , /usr/bin/aptitude\n    '
    load_path = None
    load_paths = [x for x in args if str(x).startswith('load_path=')]
    if load_paths:
        if len(load_paths) > 1:
            raise SaltInvocationError("Only one 'load_path=' value is permitted")
        else:
            load_path = load_paths[0].split('=', 1)[1]
    load_path = _check_load_paths(load_path)
    aug = _Augeas(loadpath=load_path)
    ret = {'retval': False}
    tuples = [x for x in args if not str(x).startswith('prefix=') and (not str(x).startswith('load_path='))]
    prefix = [x for x in args if str(x).startswith('prefix=')]
    if prefix:
        if len(prefix) > 1:
            raise SaltInvocationError("Only one 'prefix=' value is permitted")
        else:
            prefix = prefix[0].split('=', 1)[1]
    if len(tuples) % 2 != 0:
        raise SaltInvocationError('Uneven number of path/value arguments')
    tuple_iter = iter(tuples)
    for (path, value) in zip(tuple_iter, tuple_iter):
        target_path = path
        if prefix:
            target_path = os.path.join(prefix.rstrip('/'), path.lstrip('/'))
        try:
            aug.set(target_path, str(value))
        except ValueError as err:
            ret['error'] = 'Multiple values: {}'.format(err)
    try:
        aug.save()
        ret['retval'] = True
    except OSError as err:
        ret['error'] = str(err)
    return ret

def match(path, value='', load_path=None):
    if False:
        return 10
    "\n    Get matches for path expression\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' augeas.match /files/etc/services/service-name ssh\n\n    path\n        The path to match\n\n    value\n        The value to match on\n\n    .. versionadded:: 2016.3.0\n\n    load_path\n        A colon-spearated list of directories that modules should be searched\n        in. This is in addition to the standard load path and the directories\n        in AUGEAS_LENS_LIB.\n    "
    load_path = _check_load_paths(load_path)
    aug = _Augeas(loadpath=load_path)
    ret = {}
    try:
        matches = aug.match(path)
    except RuntimeError:
        return ret
    for _match in matches:
        if value and aug.get(_match) == value:
            ret[_match] = value
        elif not value:
            ret[_match] = aug.get(_match)
    return ret

def remove(path, load_path=None):
    if False:
        return 10
    "\n    Get matches for path expression\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' augeas.remove \\\n        /files/etc/sysctl.conf/net.ipv4.conf.all.log_martians\n\n    path\n        The path to remove\n\n    .. versionadded:: 2016.3.0\n\n    load_path\n        A colon-spearated list of directories that modules should be searched\n        in. This is in addition to the standard load path and the directories\n        in AUGEAS_LENS_LIB.\n    "
    load_path = _check_load_paths(load_path)
    aug = _Augeas(loadpath=load_path)
    ret = {'retval': False}
    try:
        count = aug.remove(path)
        aug.save()
        if count == -1:
            ret['error'] = 'Invalid node'
        else:
            ret['retval'] = True
    except (RuntimeError, OSError) as err:
        ret['error'] = str(err)
    ret['count'] = count
    return ret

def ls(path, load_path=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    List the direct children of a node\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' augeas.ls /files/etc/passwd\n\n    path\n        The path to list\n\n    .. versionadded:: 2016.3.0\n\n    load_path\n        A colon-spearated list of directories that modules should be searched\n        in. This is in addition to the standard load path and the directories\n        in AUGEAS_LENS_LIB.\n    "

    def _match(path):
        if False:
            while True:
                i = 10
        'Internal match function'
        try:
            matches = aug.match(salt.utils.stringutils.to_str(path))
        except RuntimeError:
            return {}
        ret = {}
        for _ma in matches:
            ret[_ma] = aug.get(_ma)
        return ret
    load_path = _check_load_paths(load_path)
    aug = _Augeas(loadpath=load_path)
    path = path.rstrip('/') + '/'
    match_path = path + '*'
    matches = _match(match_path)
    ret = {}
    for (key, value) in matches.items():
        name = _lstrip_word(key, path)
        if _match(key + '/*'):
            ret[name + '/'] = value
        else:
            ret[name] = value
    return ret

def tree(path, load_path=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns recursively the complete tree of a node\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' augeas.tree /files/etc/\n\n    path\n        The base of the recursive listing\n\n    .. versionadded:: 2016.3.0\n\n    load_path\n        A colon-spearated list of directories that modules should be searched\n        in. This is in addition to the standard load path and the directories\n        in AUGEAS_LENS_LIB.\n    "
    load_path = _check_load_paths(load_path)
    aug = _Augeas(loadpath=load_path)
    path = path.rstrip('/') + '/'
    match_path = path
    return dict([i for i in _recurmatch(match_path, aug)])