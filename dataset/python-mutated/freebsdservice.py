"""
The service module for FreeBSD

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import fnmatch
import logging
import os
import re
import salt.utils.decorators as decorators
import salt.utils.files
import salt.utils.path
from salt.exceptions import CommandNotFoundError
__func_alias__ = {'reload_': 'reload'}
log = logging.getLogger(__name__)
__virtualname__ = 'service'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on FreeBSD\n    '
    if __grains__['os'] == 'FreeBSD':
        return __virtualname__
    return (False, 'The freebsdservice execution module cannot be loaded: only available on FreeBSD systems.')

@decorators.memoize
def _cmd(jail=None):
    if False:
        return 10
    '\n    Return full path to service command\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) keyword argument in kwargs\n    '
    service = salt.utils.path.which('service')
    if not service:
        raise CommandNotFoundError("'service' command not found")
    if jail:
        jexec = salt.utils.path.which('jexec')
        if not jexec:
            raise CommandNotFoundError("'jexec' command not found")
        service = '{} {} {}'.format(jexec, jail, service)
    return service

def _get_jail_path(jail):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.4\n\n    Return the jail's root directory (path) as shown in jls\n\n    jail\n        The jid or jail name\n    "
    jls = salt.utils.path.which('jls')
    if not jls:
        raise CommandNotFoundError("'jls' command not found")
    jails = __salt__['cmd.run_stdout']('{} -n jid name path'.format(jls))
    for j in jails.splitlines():
        (jid, jname, path) = (x.split('=')[1].strip() for x in j.split())
        if jid == jail or jname == jail:
            return path.rstrip('/')
    return ''

def _get_rcscript(name, jail=None):
    if False:
        while True:
            i = 10
    '\n    Return full path to service rc script\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) keyword argument in kwargs\n    '
    cmd = '{} -r'.format(_cmd(jail))
    prf = _get_jail_path(jail) if jail else ''
    for line in __salt__['cmd.run_stdout'](cmd, python_shell=False).splitlines():
        if line.endswith('{}{}'.format(os.path.sep, name)):
            return os.path.join(prf, line.lstrip(os.path.sep))
    return None

def _get_rcvar(name, jail=None):
    if False:
        return 10
    '\n    Return rcvar\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) keyword argument in kwargs\n    '
    if not available(name, jail):
        log.error('Service %s not found', name)
        return False
    cmd = '{} {} rcvar'.format(_cmd(jail), name)
    for line in __salt__['cmd.run_stdout'](cmd, python_shell=False).splitlines():
        if '_enable="' not in line:
            continue
        (rcvar, _) = line.split('=', 1)
        return rcvar
    return None

def get_enabled(jail=None):
    if False:
        print('Hello World!')
    "\n    Return what services are set to run on boot\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) keyword argument in kwargs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    ret = []
    service = _cmd(jail)
    prf = _get_jail_path(jail) if jail else ''
    for svc in __salt__['cmd.run']('{} -e'.format(service)).splitlines():
        ret.append(os.path.basename(svc))
    for svc in get_all(jail):
        if svc in ret:
            continue
        if not os.path.exists('{}/etc/rc.conf.d/{}'.format(prf, svc)):
            continue
        if enabled(svc, jail=jail):
            ret.append(svc)
    return sorted(ret)

def get_disabled(jail=None):
    if False:
        while True:
            i = 10
    "\n    Return what services are available but not enabled to start at boot\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) keyword argument in kwargs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    en_ = get_enabled(jail)
    all_ = get_all(jail)
    return sorted(set(all_) - set(en_))

def _switch(name, on, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Switch on/off service start at boot.\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) and chroot keyword argument\n    in kwargs. chroot should be used when jail's /etc is mounted read-only and\n    should point to a root directory where jail's /etc is mounted read-write.\n    "
    jail = kwargs.get('jail', '')
    chroot = kwargs.get('chroot', '').rstrip('/')
    if not available(name, jail):
        return False
    rcvar = _get_rcvar(name, jail)
    if not rcvar:
        log.error('rcvar for service %s not found', name)
        return False
    if jail and (not chroot):
        chroot = _get_jail_path(jail)
    config = kwargs.get('config', __salt__['config.option']('service.config', default='{}/etc/rc.conf'.format(chroot)))
    if not config:
        rcdir = '{}/etc/rc.conf.d'.format(chroot)
        if not os.path.exists(rcdir) or not os.path.isdir(rcdir):
            log.error('%s not exists', rcdir)
            return False
        config = os.path.join(rcdir, rcvar.replace('_enable', ''))
    nlines = []
    edited = False
    if on:
        val = 'YES'
    else:
        val = 'NO'
    if os.path.exists(config):
        with salt.utils.files.fopen(config, 'r') as ifile:
            for line in ifile:
                line = salt.utils.stringutils.to_unicode(line)
                if not line.startswith('{}='.format(rcvar)):
                    nlines.append(line)
                    continue
                rest = line[len(line.split()[0]):]
                nlines.append('{}="{}"{}'.format(rcvar, val, rest))
                edited = True
    if not edited:
        if len(nlines) > 1 and nlines[-1][-1] != '\n':
            nlines[-1] = '{}\n'.format(nlines[-1])
        nlines.append('{}="{}"\n'.format(rcvar, val))
    with salt.utils.files.fopen(config, 'w') as ofile:
        nlines = [salt.utils.stringutils.to_str(_l) for _l in nlines]
        ofile.writelines(nlines)
    return True

def enable(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Enable the named service to start at boot\n\n    name\n        service name\n\n    config : /etc/rc.conf\n        Config file for managing service. If config value is\n        empty string, then /etc/rc.conf.d/<service> used.\n        See man rc.conf(5) for details.\n\n        Also service.config variable can be used to change default.\n\n    .. versionchanged:: 2016.3.4\n\n    jail (optional keyword argument)\n        the jail's id or name\n\n    chroot (optional keyword argument)\n        the jail's chroot, if the jail's /etc is not mounted read-write\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    return _switch(name, True, **kwargs)

def disable(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Disable the named service to start at boot\n\n    Arguments the same as for enable()\n\n    .. versionchanged:: 2016.3.4\n\n    jail (optional keyword argument)\n        the jail's id or name\n\n    chroot (optional keyword argument)\n        the jail's chroot, if the jail's /etc is not mounted read-write\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    return _switch(name, False, **kwargs)

def enabled(name, **kwargs):
    if False:
        return 10
    "\n    Return True if the named service is enabled, false otherwise\n\n    name\n        Service name\n\n    .. versionchanged:: 2016.3.4\n\n    Support for jail (representing jid or jail name) keyword argument in kwargs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    jail = kwargs.get('jail', '')
    if not available(name, jail):
        log.error('Service %s not found', name)
        return False
    cmd = '{} {} rcvar'.format(_cmd(jail), name)
    for line in __salt__['cmd.run_stdout'](cmd, python_shell=False).splitlines():
        if '_enable="' not in line:
            continue
        (_, state, _) = line.split('"', 2)
        return state.lower() in ('yes', 'true', 'on', '1')
    return False

def disabled(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    return not enabled(name, **kwargs)

def available(name, jail=None):
    if False:
        return 10
    "\n    Check that the given service is available.\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    return name in get_all(jail)

def missing(name, jail=None):
    if False:
        return 10
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return name not in get_all(jail)

def get_all(jail=None):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of all available services\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    ret = []
    service = _cmd(jail)
    for srv in __salt__['cmd.run']('{} -l'.format(service)).splitlines():
        if not srv.isupper():
            ret.append(srv)
    return sorted(ret)

def start(name, jail=None):
    if False:
        i = 10
        return i + 15
    "\n    Start the specified service\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = '{} {} onestart'.format(_cmd(jail), name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def stop(name, jail=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop the specified service\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = '{} {} onestop'.format(_cmd(jail), name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def restart(name, jail=None):
    if False:
        i = 10
        return i + 15
    "\n    Restart the named service\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = '{} {} onerestart'.format(_cmd(jail), name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def reload_(name, jail=None):
    if False:
        while True:
            i = 10
    "\n    Restart the named service\n\n    .. versionchanged:: 2016.3.4\n\n    jail: optional jid or jail name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = '{} {} onereload'.format(_cmd(jail), name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def status(name, sig=None, jail=None):
    if False:
        while True:
            i = 10
    "\n    Return the status for a service.\n    If the name contains globbing, a dict mapping service name to True/False\n    values is returned.\n\n    .. versionchanged:: 2016.3.4\n\n    .. versionchanged:: 2018.3.0\n        The service name can now be a glob (e.g. ``salt*``)\n\n    Args:\n        name (str): The name of the service to check\n        sig (str): Signature to use to find the service via ps\n\n    Returns:\n        bool: True if running, False otherwise\n        dict: Maps service name to True if running, False otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name> [service signature]\n    "
    if sig:
        return bool(__salt__['status.pid'](sig))
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        cmd = '{} {} onestatus'.format(_cmd(jail), service)
        results[service] = not __salt__['cmd.retcode'](cmd, python_shell=False, ignore_retcode=True)
    if contains_globbing:
        return results
    return results[name]