"""
Support for poudriere
"""
import logging
import os
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Module load on freebsd only and if poudriere installed\n    '
    if __grains__['os'] == 'FreeBSD' and salt.utils.path.which('poudriere'):
        return 'poudriere'
    else:
        return (False, 'The poudriere execution module failed to load: only available on FreeBSD with the poudriere binary in the path.')

def _config_file():
    if False:
        return 10
    '\n    Return the config file location to use\n    '
    return __salt__['config.option']('poudriere.config')

def _config_dir():
    if False:
        i = 10
        return i + 15
    '\n    Return the configuration directory to use\n    '
    return __salt__['config.option']('poudriere.config_dir')

def _check_config_exists(config_file=None):
    if False:
        i = 10
        return i + 15
    '\n    Verify the config file is present\n    '
    if config_file is None:
        config_file = _config_file()
    if not os.path.isfile(config_file):
        return False
    return True

def is_jail(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return True if jail exists False if not\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.is_jail <jail name>\n    "
    jails = list_jails()
    for jail in jails:
        if jail.split()[0] == name:
            return True
    return False

def make_pkgng_aware(jname):
    if False:
        while True:
            i = 10
    "\n    Make jail ``jname`` pkgng aware\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.make_pkgng_aware <jail name>\n    "
    ret = {'changes': {}}
    cdir = _config_dir()
    if not os.path.isdir(cdir):
        os.makedirs(cdir)
        if os.path.isdir(cdir):
            ret['changes'] = 'Created poudriere make file dir {}'.format(cdir)
        else:
            return 'Could not create or find required directory {}'.format(cdir)
    __salt__['file.write']('{}-make.conf'.format(os.path.join(cdir, jname)), 'WITH_PKGNG=yes')
    if os.path.isfile(os.path.join(cdir, jname) + '-make.conf'):
        ret['changes'] = 'Created {}'.format(os.path.join(cdir, '{}-make.conf'.format(jname)))
        return ret
    else:
        return 'Looks like file {} could not be created'.format(os.path.join(cdir, jname + '-make.conf'))

def parse_config(config_file=None):
    if False:
        print('Hello World!')
    "\n    Returns a dict of poudriere main configuration definitions\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.parse_config\n    "
    if config_file is None:
        config_file = _config_file()
    ret = {}
    if _check_config_exists(config_file):
        with salt.utils.files.fopen(config_file) as ifile:
            for line in ifile:
                (key, val) = salt.utils.stringutils.to_unicode(line).split('=')
                ret[key] = val
        return ret
    return 'Could not find {} on file system'.format(config_file)

def version():
    if False:
        print('Hello World!')
    "\n    Return poudriere version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.version\n    "
    cmd = 'poudriere version'
    return __salt__['cmd.run'](cmd)

def list_jails():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of current jails managed by poudriere\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.list_jails\n    "
    _check_config_exists()
    cmd = 'poudriere jails -l'
    res = __salt__['cmd.run'](cmd)
    return res.splitlines()

def list_ports():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of current port trees managed by poudriere\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.list_ports\n    "
    _check_config_exists()
    cmd = 'poudriere ports -l'
    res = __salt__['cmd.run'](cmd).splitlines()
    return res

def create_jail(name, arch, version='9.0-RELEASE'):
    if False:
        return 10
    "\n    Creates a new poudriere jail if one does not exist\n\n    *NOTE* creating a new jail will take some time the master is not hanging\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.create_jail 90amd64 amd64\n    "
    _check_config_exists()
    if is_jail(name):
        return '{} already exists'.format(name)
    cmd = 'poudriere jails -c -j {} -v {} -a {}'.format(name, version, arch)
    __salt__['cmd.run'](cmd)
    make_pkgng_aware(name)
    if is_jail(name):
        return 'Created jail {}'.format(name)
    return 'Issue creating jail {}'.format(name)

def update_jail(name):
    if False:
        i = 10
        return i + 15
    "\n    Run freebsd-update on `name` poudriere jail\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.update_jail freebsd:10:x86:64\n    "
    if is_jail(name):
        cmd = 'poudriere jail -u -j {}'.format(name)
        ret = __salt__['cmd.run'](cmd)
        return ret
    else:
        return 'Could not find jail {}'.format(name)

def delete_jail(name):
    if False:
        while True:
            i = 10
    "\n    Deletes poudriere jail with `name`\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.delete_jail 90amd64\n    "
    if is_jail(name):
        cmd = 'poudriere jail -d -j {}'.format(name)
        __salt__['cmd.run'](cmd)
        if is_jail(name):
            return 'Looks like there was an issue deleting jail {}'.format(name)
    else:
        return 'Looks like jail {} has not been created'.format(name)
    make_file = os.path.join(_config_dir(), '{}-make.conf'.format(name))
    if os.path.isfile(make_file):
        try:
            os.remove(make_file)
        except OSError:
            return 'Deleted jail "{}" but was unable to remove jail make file'.format(name)
        __salt__['file.remove'](make_file)
    return 'Deleted jail {}'.format(name)

def info_jail(name):
    if False:
        while True:
            i = 10
    "\n    Show information on `name` poudriere jail\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.info_jail head-amd64\n    "
    if is_jail(name):
        cmd = 'poudriere jail -i -j {}'.format(name)
        ret = __salt__['cmd.run'](cmd).splitlines()
        return ret
    else:
        return 'Could not find jail {}'.format(name)

def create_ports_tree():
    if False:
        i = 10
        return i + 15
    '\n    Not working need to run portfetch non interactive\n    '
    _check_config_exists()
    cmd = 'poudriere ports -c'
    ret = __salt__['cmd.run'](cmd)
    return ret

def update_ports_tree(ports_tree):
    if False:
        print('Hello World!')
    "\n    Updates the ports tree, either the default or the `ports_tree` specified\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' poudriere.update_ports_tree staging\n    "
    _check_config_exists()
    if ports_tree:
        cmd = 'poudriere ports -u -p {}'.format(ports_tree)
    else:
        cmd = 'poudriere ports -u'
    ret = __salt__['cmd.run'](cmd)
    return ret

def bulk_build(jail, pkg_file, keep=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run bulk build on poudriere server.\n\n    Return number of pkg builds, failures, and errors, on error dump to CLI\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt -N buildbox_group poudriere.bulk_build 90amd64 /root/pkg_list\n\n    '
    if not os.path.isfile(pkg_file):
        return 'Could not find file {} on filesystem'.format(pkg_file)
    if not is_jail(jail):
        return 'Could not find jail {}'.format(jail)
    if keep:
        cmd = 'poudriere bulk -k -f {} -j {}'.format(pkg_file, jail)
    else:
        cmd = 'poudriere bulk -f {} -j {}'.format(pkg_file, jail)
    res = __salt__['cmd.run'](cmd)
    lines = res.splitlines()
    for line in lines:
        if 'packages built' in line:
            return line
    return 'There may have been an issue building packages dumping output: {}'.format(res)