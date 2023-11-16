"""
Manage Perl modules using CPAN

.. versionadded:: 2015.5.0
"""
import logging
import os
import os.path
import salt.utils.files
import salt.utils.path
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on supported POSIX-like systems\n    '
    if salt.utils.path.which('cpan'):
        return True
    return (False, 'Unable to locate cpan. Make sure it is installed and in the PATH.')

def install(module):
    if False:
        return 10
    "\n    Install a Perl module from CPAN\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cpan.install Template::Alloy\n    "
    ret = {'old': None, 'new': None}
    old_info = show(module)
    cmd = 'cpan -i {}'.format(module)
    out = __salt__['cmd.run'](cmd)
    if "don't know what it is" in out:
        ret['error'] = 'CPAN cannot identify this package'
        return ret
    new_info = show(module)
    ret['old'] = old_info.get('installed version', None)
    ret['new'] = new_info['installed version']
    return ret

def remove(module, details=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attempt to remove a Perl module that was installed from CPAN. Because the\n    ``cpan`` command doesn\'t actually support "uninstall"-like functionality,\n    this function will attempt to do what it can, with what it has from CPAN.\n\n    Until this function is declared stable, USE AT YOUR OWN RISK!\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' cpan.remove Old::Package\n    '
    ret = {'old': None, 'new': None}
    info = show(module)
    if 'error' in info:
        return {'error': info['error']}
    version = info.get('installed version', None)
    if version is None:
        return ret
    ret['old'] = version
    if 'cpan build dirs' not in info:
        return {'error': 'No CPAN data available to use for uninstalling'}
    mod_pathfile = module.replace('::', '/') + '.pm'
    ins_path = info['installed file'].replace(mod_pathfile, '')
    files = []
    for build_dir in info['cpan build dirs']:
        contents = os.listdir(build_dir)
        if 'MANIFEST' not in contents:
            continue
        mfile = os.path.join(build_dir, 'MANIFEST')
        with salt.utils.files.fopen(mfile, 'r') as fh_:
            for line in fh_.readlines():
                line = salt.utils.stringutils.to_unicode(line)
                if line.startswith('lib/'):
                    files.append(line.replace('lib/', ins_path).strip())
    rm_details = {}
    for file_ in files:
        if file_ in rm_details:
            continue
        log.trace('Removing %s', file_)
        if __salt__['file.remove'](file_):
            rm_details[file_] = 'removed'
        else:
            rm_details[file_] = 'unable to remove'
    if details:
        ret['details'] = rm_details
    return ret

def list_():
    if False:
        for i in range(10):
            print('nop')
    "\n    List installed Perl modules, and the version installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cpan.list\n    "
    ret = {}
    cmd = 'cpan -l'
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        comps = line.split()
        ret[comps[0]] = comps[1]
    return ret

def show(module):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show information about a specific Perl module\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cpan.show Template::Alloy\n    "
    ret = {}
    ret['name'] = module
    cmd = 'cpan -D {}'.format(module)
    out = __salt__['cmd.run'](cmd).splitlines()
    mode = 'skip'
    info = []
    for line in out:
        if line.startswith('-------------'):
            mode = 'parse'
            continue
        if mode == 'skip':
            continue
        info.append(line)
    if len(info) == 6:
        info.insert(2, '')
    if len(info) < 6:
        ret['error'] = 'This package does not seem to exist'
        return ret
    ret['description'] = info[0].strip()
    ret['cpan file'] = info[1].strip()
    if info[2].strip():
        ret['installed file'] = info[2].strip()
    else:
        ret['installed file'] = None
    comps = info[3].split(':')
    if len(comps) > 1:
        ret['installed version'] = comps[1].strip()
    if 'installed version' not in ret or not ret['installed version']:
        ret['installed version'] = None
    comps = info[4].split(':')
    comps = comps[1].split()
    ret['cpan version'] = comps[0].strip()
    ret['author name'] = info[5].strip()
    ret['author email'] = info[6].strip()
    config = show_config()
    build_dir = config.get('build_dir', None)
    if build_dir is not None:
        ret['cpan build dirs'] = []
        builds = os.listdir(build_dir)
        pfile = module.replace('::', '-')
        for file_ in builds:
            if file_.startswith(pfile):
                ret['cpan build dirs'].append(os.path.join(build_dir, file_))
    return ret

def show_config():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a dict of CPAN configuration values\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cpan.show_config\n    "
    ret = {}
    cmd = 'cpan -J'
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        if '=>' not in line:
            continue
        comps = line.split('=>')
        key = comps[0].replace("'", '').strip()
        val = comps[1].replace("',", '').replace("'", '').strip()
        ret[key] = val
    return ret