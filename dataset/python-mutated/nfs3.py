"""
Module for managing NFS version 3.
"""
import logging
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work on POSIX-like systems\n    '
    if not salt.utils.path.which('showmount'):
        return (False, 'The nfs3 execution module failed to load: the showmount binary is not in the path.')
    return True

def list_exports(exports='/etc/exports'):
    if False:
        for i in range(10):
            print('nop')
    "\n    List configured exports\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nfs.list_exports\n    "
    ret = {}
    with salt.utils.files.fopen(exports, 'r') as efl:
        for line in salt.utils.stringutils.to_unicode(efl.read()).splitlines():
            if not line:
                continue
            if line.startswith('#'):
                continue
            comps = line.split()
            if not comps[0] in ret:
                ret[comps[0]] = []
            newshares = []
            for perm in comps[1:]:
                if perm.startswith('/'):
                    newshares.append(perm)
                    continue
                permcomps = perm.split('(')
                permcomps[1] = permcomps[1].replace(')', '')
                hosts = permcomps[0]
                if not isinstance(hosts, str):
                    raise TypeError('hosts argument must be a string')
                options = permcomps[1].split(',')
                ret[comps[0]].append({'hosts': hosts, 'options': options})
            for share in newshares:
                ret[share] = ret[comps[0]]
    return ret

def del_export(exports='/etc/exports', path=None):
    if False:
        print('Hello World!')
    "\n    Remove an export\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nfs.del_export /media/storage\n    "
    edict = list_exports(exports)
    del edict[path]
    _write_exports(exports, edict)
    return edict

def add_export(exports='/etc/exports', path=None, hosts=None, options=None):
    if False:
        print('Hello World!')
    "\n    Add an export\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nfs3.add_export path='/srv/test' hosts='127.0.0.1' options=['rw']\n    "
    if options is None:
        options = []
    if not isinstance(hosts, str):
        raise TypeError('hosts argument must be a string')
    edict = list_exports(exports)
    if path not in edict:
        edict[path] = []
    new = {'hosts': hosts, 'options': options}
    edict[path].append(new)
    _write_exports(exports, edict)
    return new

def _write_exports(exports, edict):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write an exports file to disk\n\n    If multiple shares were initially configured per line, like:\n\n        /media/storage /media/data *(ro,sync,no_subtree_check)\n\n    ...then they will be saved to disk with only one share per line:\n\n        /media/storage *(ro,sync,no_subtree_check)\n        /media/data *(ro,sync,no_subtree_check)\n    '
    with salt.utils.files.fopen(exports, 'w') as efh:
        for export in edict:
            line = salt.utils.stringutils.to_str(export)
            for perms in edict[export]:
                hosts = perms['hosts']
                options = ','.join(perms['options'])
                line += ' {}({})'.format(hosts, options)
            efh.write('{}\n'.format(line))

def reload_exports():
    if False:
        i = 10
        return i + 15
    "\n    Trigger a reload of the exports file to apply changes\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' nfs3.reload_exports\n    "
    ret = {}
    command = 'exportfs -r'
    output = __salt__['cmd.run_all'](command)
    ret['stdout'] = output['stdout']
    ret['stderr'] = output['stderr']
    ret['result'] = output['stderr'] == ''
    return ret