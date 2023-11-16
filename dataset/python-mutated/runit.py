"""
runit service module
(http://smarden.org/runit)

This module is compatible with the :mod:`service <salt.states.service>` states,
so it can be used to maintain services using the ``provider`` argument:

.. code-block:: yaml

    myservice:
      service:
        - running
        - provider: runit

Provides virtual `service` module on systems using runit as init.


Service management rules (`sv` command):

    service $n is ENABLED   if file SERVICE_DIR/$n/run exists
    service $n is AVAILABLE if ENABLED or if file AVAIL_SVR_DIR/$n/run exists
    service $n is DISABLED  if AVAILABLE but not ENABLED

    SERVICE_DIR/$n is normally a symlink to a AVAIL_SVR_DIR/$n folder


Service auto-start/stop mechanism:

    `sv` (auto)starts/stops service as soon as SERVICE_DIR/<service> is
    created/deleted, both on service creation or a boot time.

    autostart feature is disabled if file SERVICE_DIR/<n>/down exists. This
    does not affect the current's service status (if already running) nor
    manual service management.


Service's alias:

    Service `sva` is an alias of service `svc` when `AVAIL_SVR_DIR/sva` symlinks
    to folder `AVAIL_SVR_DIR/svc`. `svc` can't be enabled if it is already
    enabled through an alias already enabled, since `sv` files are stored in
    folder `SERVICE_DIR/svc/`.

    XBPS package management uses a service's alias to provides service
    alternative(s), such as chrony and openntpd both aliased to ntpd.
"""
import glob
import logging
import os
import time
import salt.utils.files
import salt.utils.path
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__func_alias__ = {'reload_': 'reload'}
VALID_SERVICE_DIRS = ['/service', '/var/service', '/etc/service']
SERVICE_DIR = None
for service_dir in VALID_SERVICE_DIRS:
    if os.path.exists(service_dir):
        SERVICE_DIR = service_dir
        break
AVAIL_SVR_DIRS = []
__virtualname__ = 'runit'
__virtual_aliases__ = ('runit',)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Virtual service only on systems using runit as init process (PID 1).\n    Otherwise, use this module with the provider mechanism.\n    '
    if __grains__.get('init') == 'runit':
        if __grains__['os'] == 'Void':
            add_svc_avail_path('/etc/sv')
        global __virtualname__
        __virtualname__ = 'service'
        return __virtualname__
    if salt.utils.path.which('sv'):
        return __virtualname__
    return (False, 'Runit not available.  Please install sv')

def _service_path(name):
    if False:
        print('Hello World!')
    "\n    Return SERVICE_DIR+name if possible\n\n    name\n        the service's name to work on\n    "
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    return os.path.join(SERVICE_DIR, name)

def start(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start service\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.start <service name>\n    "
    cmd = f'sv start {_service_path(name)}'
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        i = 10
        return i + 15
    "\n    Stop service\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.stop <service name>\n    "
    cmd = f'sv stop {_service_path(name)}'
    return not __salt__['cmd.retcode'](cmd)

def reload_(name):
    if False:
        i = 10
        return i + 15
    "\n    Reload service\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.reload <service name>\n    "
    cmd = f'sv reload {_service_path(name)}'
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        i = 10
        return i + 15
    "\n    Restart service\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.restart <service name>\n    "
    cmd = f'sv restart {_service_path(name)}'
    return not __salt__['cmd.retcode'](cmd)

def full_restart(name):
    if False:
        print('Hello World!')
    "\n    Calls runit.restart()\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.full_restart <service name>\n    "
    restart(name)

def status(name, sig=None):
    if False:
        print('Hello World!')
    "\n    Return ``True`` if service is running\n\n    name\n        the service's name\n\n    sig\n        signature to identify with ps\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.status <service name>\n    "
    if sig:
        return bool(__salt__['status.pid'](sig))
    svc_path = _service_path(name)
    if not os.path.exists(svc_path):
        return False
    cmd = f'sv status {svc_path}'
    try:
        out = __salt__['cmd.run_stdout'](cmd)
        return out.startswith('run: ')
    except Exception:
        return False

def _is_svc(svc_path):
    if False:
        return 10
    '\n    Return ``True`` if directory <svc_path> is really a service:\n    file <svc_path>/run exists and is executable\n\n    svc_path\n        the (absolute) directory to check for compatibility\n    '
    run_file = os.path.join(svc_path, 'run')
    if os.path.exists(svc_path) and os.path.exists(run_file) and os.access(run_file, os.X_OK):
        return True
    return False

def status_autostart(name):
    if False:
        i = 10
        return i + 15
    "\n    Return ``True`` if service <name> is autostarted by sv\n    (file $service_folder/down does not exist)\n    NB: return ``False`` if the service is not enabled.\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.status_autostart <service name>\n    "
    return not os.path.exists(os.path.join(_service_path(name), 'down'))

def get_svc_broken_path(name='*'):
    if False:
        print('Hello World!')
    "\n    Return list of broken path(s) in SERVICE_DIR that match ``name``\n\n    A path is broken if it is a broken symlink or can not be a runit service\n\n    name\n        a glob for service name. default is '*'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.get_svc_broken_path <service name>\n    "
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    ret = set()
    for el in glob.glob(os.path.join(SERVICE_DIR, name)):
        if not _is_svc(el):
            ret.add(el)
    return sorted(ret)

def get_svc_avail_path():
    if False:
        print('Hello World!')
    '\n    Return list of paths that may contain available services\n    '
    return AVAIL_SVR_DIRS

def add_svc_avail_path(path):
    if False:
        while True:
            i = 10
    '\n    Add a path that may contain available services.\n    Return ``True`` if added (or already present), ``False`` on error.\n\n    path\n        directory to add to AVAIL_SVR_DIRS\n    '
    if os.path.exists(path):
        if path not in AVAIL_SVR_DIRS:
            AVAIL_SVR_DIRS.append(path)
        return True
    return False

def _get_svc_path(name='*', status=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of paths to services with ``name`` that have the specified ``status``\n\n    name\n        a glob for service name. default is '*'\n\n    status\n        None       : all services (no filter, default choice)\n        'DISABLED' : available service(s) that is not enabled\n        'ENABLED'  : enabled service (whether started on boot or not)\n    "
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    ena = set()
    for el in glob.glob(os.path.join(SERVICE_DIR, name)):
        if _is_svc(el):
            if os.path.islink(el):
                ena.add(os.readlink(el))
            else:
                ena.add(el)
            log.trace('found enabled service path: %s', el)
    if status == 'ENABLED':
        return sorted(ena)
    ava = set()
    for d in AVAIL_SVR_DIRS:
        for el in glob.glob(os.path.join(d, name)):
            if _is_svc(el):
                ava.add(el)
                log.trace('found available service path: %s', el)
    if status == 'DISABLED':
        ret = ava.difference(ena)
    else:
        ret = ava.union(ena)
    return sorted(ret)

def _get_svc_list(name='*', status=None):
    if False:
        print('Hello World!')
    "\n    Return list of services that have the specified service ``status``\n\n    name\n        a glob for service name. default is '*'\n\n    status\n        None       : all services (no filter, default choice)\n        'DISABLED' : available service that is not enabled\n        'ENABLED'  : enabled service (whether started on boot or not)\n    "
    return sorted((os.path.basename(el) for el in _get_svc_path(name, status)))

def get_svc_alias():
    if False:
        while True:
            i = 10
    "\n    Returns the list of service's name that are aliased and their alias path(s)\n    "
    ret = {}
    for d in AVAIL_SVR_DIRS:
        for el in glob.glob(os.path.join(d, '*')):
            if not os.path.islink(el):
                continue
            psvc = os.readlink(el)
            if not os.path.isabs(psvc):
                psvc = os.path.join(d, psvc)
            nsvc = os.path.basename(psvc)
            if nsvc not in ret:
                ret[nsvc] = []
            ret[nsvc].append(el)
    return ret

def available(name):
    if False:
        return 10
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.available <service name>\n    "
    return name in _get_svc_list(name)

def missing(name):
    if False:
        while True:
            i = 10
    "\n    The inverse of runit.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.missing <service name>\n    "
    return name not in _get_svc_list(name)

def get_all():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of all available services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' runit.get_all\n    "
    return _get_svc_list()

def get_enabled():
    if False:
        while True:
            i = 10
    "\n    Return a list of all enabled services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    return _get_svc_list(status='ENABLED')

def get_disabled():
    if False:
        return 10
    "\n    Return a list of all disabled services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    return _get_svc_list(status='DISABLED')

def enabled(name):
    if False:
        i = 10
        return i + 15
    "\n    Return ``True`` if the named service is enabled, ``False`` otherwise\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    return name in _get_svc_list(name, 'ENABLED')

def disabled(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return ``True`` if the named service is disabled, ``False``  otherwise\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    return name not in _get_svc_list(name, 'ENABLED')

def show(name):
    if False:
        print('Hello World!')
    "\n    Show properties of one or more units/jobs or the manager\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.show <service name>\n    "
    ret = {}
    ret['enabled'] = False
    ret['disabled'] = True
    ret['running'] = False
    ret['service_path'] = None
    ret['autostart'] = False
    ret['command_path'] = None
    ret['available'] = available(name)
    if not ret['available']:
        return ret
    ret['enabled'] = enabled(name)
    ret['disabled'] = not ret['enabled']
    ret['running'] = status(name)
    ret['autostart'] = status_autostart(name)
    ret['service_path'] = _get_svc_path(name)[0]
    if ret['service_path']:
        ret['command_path'] = os.path.join(ret['service_path'], 'run')
    return ret

def enable(name, start=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start service ``name`` at boot.\n    Returns ``True`` if operation is successful\n\n    name\n        the service's name\n\n    start : False\n        If ``True``, start the service once enabled.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <name> [start=True]\n    "
    if not available(name):
        return False
    alias = get_svc_alias()
    if name in alias:
        log.error('This service is aliased, enable its alias instead')
        return False
    svc_realpath = _get_svc_path(name)[0]
    down_file = os.path.join(svc_realpath, 'down')
    if enabled(name):
        if os.path.exists(down_file):
            try:
                os.unlink(down_file)
            except OSError:
                log.error('Unable to remove file %s', down_file)
                return False
        return True
    if not start:
        log.trace('need a temporary file %s', down_file)
        if not os.path.exists(down_file):
            try:
                salt.utils.files.fopen(down_file, 'w').close()
            except OSError:
                log.error('Unable to create file %s', down_file)
                return False
    try:
        os.symlink(svc_realpath, _service_path(name))
    except OSError:
        log.error('Unable to create symlink %s', down_file)
        if not start:
            os.unlink(down_file)
        return False
    cmd = f'sv status {_service_path(name)}'
    retcode_sv = 1
    count_sv = 0
    while retcode_sv != 0 and count_sv < 10:
        time.sleep(0.5)
        count_sv += 1
        call = __salt__['cmd.run_all'](cmd)
        retcode_sv = call['retcode']
    if not start and os.path.exists(down_file):
        try:
            os.unlink(down_file)
        except OSError:
            log.error('Unable to remove temp file %s', down_file)
            retcode_sv = 1
    if retcode_sv != 0:
        os.unlink(os.path.join([_service_path(name), name]))
        return False
    return True

def disable(name, stop=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Don't start service ``name`` at boot\n    Returns ``True`` if operation is successful\n\n    name\n        the service's name\n\n    stop\n        if True, also stops the service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <name> [stop=True]\n    "
    if not enabled(name):
        return False
    svc_realpath = _get_svc_path(name)[0]
    down_file = os.path.join(svc_realpath, 'down')
    if stop:
        stop(name)
    if not os.path.exists(down_file):
        try:
            salt.utils.files.fopen(down_file, 'w').close()
        except OSError:
            log.error('Unable to create file %s', down_file)
            return False
    return True

def remove(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the service <name> from system.\n    Returns ``True`` if operation is successful.\n    The service will be also stopped.\n\n    name\n        the service's name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.remove <name>\n    "
    if not enabled(name):
        return False
    svc_path = _service_path(name)
    if not os.path.islink(svc_path):
        log.error('%s is not a symlink: not removed', svc_path)
        return False
    if not stop(name):
        log.error('Failed to stop service %s', name)
        return False
    try:
        os.remove(svc_path)
    except OSError:
        log.error('Unable to remove symlink %s', svc_path)
        return False
    return True