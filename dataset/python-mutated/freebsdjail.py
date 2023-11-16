"""
The jail module for FreeBSD
"""
import os
import re
import subprocess
import salt.utils.args
import salt.utils.files
import salt.utils.stringutils
__virtualname__ = 'jail'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only runs on FreeBSD systems\n    '
    if __grains__['os'] == 'FreeBSD':
        return __virtualname__
    return (False, 'The freebsdjail execution module cannot be loaded: only available on FreeBSD systems.')

def start(jail=''):
    if False:
        i = 10
        return i + 15
    "\n    Start the specified jail or all, if none specified\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.start [<jail name>]\n    "
    cmd = 'service jail onestart {}'.format(jail)
    return not __salt__['cmd.retcode'](cmd)

def stop(jail=''):
    if False:
        print('Hello World!')
    "\n    Stop the specified jail or all, if none specified\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.stop [<jail name>]\n    "
    cmd = 'service jail onestop {}'.format(jail)
    return not __salt__['cmd.retcode'](cmd)

def restart(jail=''):
    if False:
        return 10
    "\n    Restart the specified jail or all, if none specified\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.restart [<jail name>]\n    "
    cmd = 'service jail onerestart {}'.format(jail)
    return not __salt__['cmd.retcode'](cmd)

def is_enabled():
    if False:
        while True:
            i = 10
    "\n    See if jail service is actually enabled on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.is_enabled <jail name>\n    "
    cmd = 'service -e'
    services = __salt__['cmd.run'](cmd, python_shell=False)
    for service in services.split('\\n'):
        if re.search('jail', service):
            return True
    return False

def get_enabled():
    if False:
        print('Hello World!')
    "\n    Return which jails are set to be run\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.get_enabled\n    "
    ret = []
    for rconf in ('/etc/rc.conf', '/etc/rc.conf.local'):
        if os.access(rconf, os.R_OK):
            with salt.utils.files.fopen(rconf, 'r') as _fp:
                for line in _fp:
                    line = salt.utils.stringutils.to_unicode(line)
                    if not line.strip():
                        continue
                    if not line.startswith('jail_list='):
                        continue
                    jails = line.split('"')[1].split()
                    for j in jails:
                        ret.append(j)
    return ret

def show_config(jail):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display specified jail's configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.show_config <jail name>\n    "
    ret = {}
    if subprocess.call(['jls', '-nq', '-j', jail]) == 0:
        jls = subprocess.check_output(['jls', '-nq', '-j', jail])
        jailopts = salt.utils.args.shlex_split(salt.utils.stringutils.to_unicode(jls))
        for jailopt in jailopts:
            if '=' not in jailopt:
                ret[jailopt.strip().rstrip(';')] = '1'
            else:
                key = jailopt.split('=')[0].strip()
                value = jailopt.split('=')[-1].strip().strip('"')
                ret[key] = value
    else:
        for rconf in ('/etc/rc.conf', '/etc/rc.conf.local'):
            if os.access(rconf, os.R_OK):
                with salt.utils.files.fopen(rconf, 'r') as _fp:
                    for line in _fp:
                        line = salt.utils.stringutils.to_unicode(line)
                        if not line.strip():
                            continue
                        if not line.startswith('jail_{}_'.format(jail)):
                            continue
                        (key, value) = line.split('=')
                        ret[key.split('_', 2)[2]] = value.split('"')[1]
        for jconf in ('/etc/jail.conf', '/usr/local/etc/jail.conf'):
            if os.access(jconf, os.R_OK):
                with salt.utils.files.fopen(jconf, 'r') as _fp:
                    for line in _fp:
                        line = salt.utils.stringutils.to_unicode(line)
                        line = line.partition('#')[0].strip()
                        if line:
                            if line.split()[-1] == '{':
                                if line.split()[0] != jail and line.split()[0] != '*':
                                    while line.split()[-1] != '}':
                                        line = next(_fp)
                                        line = line.partition('#')[0].strip()
                                else:
                                    continue
                            if line.split()[-1] == '}':
                                continue
                            if '=' not in line:
                                ret[line.strip().rstrip(';')] = '1'
                            else:
                                key = line.split('=')[0].strip()
                                value = line.split('=')[-1].strip().strip(';\'"')
                                ret[key] = value
    return ret

def fstab(jail):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display contents of a fstab(5) file defined in specified\n    jail's configuration. If no file is defined, return False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.fstab <jail name>\n    "
    ret = []
    config = show_config(jail)
    if 'fstab' in config:
        c_fstab = config['fstab']
    elif 'mount.fstab' in config:
        c_fstab = config['mount.fstab']
    if 'fstab' in config or 'mount.fstab' in config:
        if os.access(c_fstab, os.R_OK):
            with salt.utils.files.fopen(c_fstab, 'r') as _fp:
                for line in _fp:
                    line = salt.utils.stringutils.to_unicode(line)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    try:
                        (device, mpoint, fstype, opts, dump, pas_) = line.split()
                    except ValueError:
                        continue
                    ret.append({'device': device, 'mountpoint': mpoint, 'fstype': fstype, 'options': opts, 'dump': dump, 'pass': pas_})
    if not ret:
        ret = False
    return ret

def status(jail):
    if False:
        return 10
    "\n    See if specified jail is currently running\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.status <jail name>\n    "
    cmd = 'jls'
    found_jails = __salt__['cmd.run'](cmd, python_shell=False)
    for found_jail in found_jails.split('\\n'):
        if re.search(jail, found_jail):
            return True
    return False

def sysctl():
    if False:
        while True:
            i = 10
    "\n    Dump all jail related kernel states (sysctl)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jail.sysctl\n    "
    ret = {}
    sysctl_jail = __salt__['cmd.run']('sysctl security.jail')
    for line in sysctl_jail.splitlines():
        (key, value) = line.split(':', 1)
        ret[key.strip()] = value.strip()
    return ret