"""
Module for gathering and managing information about MooseFS
"""
import salt.utils.path

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the mfs commands are installed\n    '
    if salt.utils.path.which('mfsgetgoal'):
        return 'moosefs'
    return (False, 'The moosefs execution module cannot be loaded: the mfsgetgoal binary is not in the path.')

def dirinfo(path, opts=None):
    if False:
        i = 10
        return i + 15
    "\n    Return information on a directory located on the Moose\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' moosefs.dirinfo /path/to/dir/ [-[n][h|H]]\n    "
    cmd = 'mfsdirinfo'
    ret = {}
    if opts:
        cmd += ' -' + opts
    cmd += ' ' + path
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    output = out['stdout'].splitlines()
    for line in output:
        if not line:
            continue
        comps = line.split(':')
        ret[comps[0].strip()] = comps[1].strip()
    return ret

def fileinfo(path):
    if False:
        i = 10
        return i + 15
    "\n    Return information on a file located on the Moose\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' moosefs.fileinfo /path/to/dir/\n    "
    cmd = 'mfsfileinfo ' + path
    ret = {}
    chunknum = ''
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    output = out['stdout'].splitlines()
    for line in output:
        if not line:
            continue
        if '/' in line:
            comps = line.split('/')
            chunknum = comps[0].strip().split(':')
            meta = comps[1].strip().split(' ')
            chunk = chunknum[0].replace('chunk ', '')
            loc = chunknum[1].strip()
            id_ = meta[0].replace('(id:', '')
            ver = meta[1].replace(')', '').replace('ver:', '')
            ret[chunknum[0]] = {'chunk': chunk, 'loc': loc, 'id': id_, 'ver': ver}
        if 'copy' in line:
            copyinfo = line.strip().split(':')
            ret[chunknum[0]][copyinfo[0]] = {'copy': copyinfo[0].replace('copy ', ''), 'ip': copyinfo[1].strip(), 'port': copyinfo[2]}
    return ret

def mounts():
    if False:
        while True:
            i = 10
    "\n    Return a list of current MooseFS mounts\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' moosefs.mounts\n    "
    cmd = 'mount'
    ret = {}
    out = __salt__['cmd.run_all'](cmd)
    output = out['stdout'].splitlines()
    for line in output:
        if not line:
            continue
        if 'fuse.mfs' in line:
            comps = line.split(' ')
            info1 = comps[0].split(':')
            info2 = info1[1].split('/')
            ret[comps[2]] = {'remote': {'master': info1[0], 'port': info2[0], 'subfolder': '/' + info2[1]}, 'local': comps[2], 'options': comps[5].replace('(', '').replace(')', '').split(',')}
    return ret

def getgoal(path, opts=None):
    if False:
        return 10
    "\n    Return goal(s) for a file or directory\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' moosefs.getgoal /path/to/file [-[n][h|H]]\n        salt '*' moosefs.getgoal /path/to/dir/ [-[n][h|H][r]]\n    "
    cmd = 'mfsgetgoal'
    ret = {}
    if opts:
        cmd += ' -' + opts
    else:
        opts = ''
    cmd += ' ' + path
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    output = out['stdout'].splitlines()
    if 'r' not in opts:
        goal = output[0].split(': ')
        ret = {'goal': goal[1]}
    else:
        for line in output:
            if not line:
                continue
            if path in line:
                continue
            comps = line.split()
            keytext = comps[0] + ' with goal'
            if keytext not in ret:
                ret[keytext] = {}
            ret[keytext][comps[3]] = comps[5]
    return ret