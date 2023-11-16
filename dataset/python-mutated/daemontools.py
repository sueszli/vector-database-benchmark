"""
daemontools service module. This module will create daemontools type
service watcher.

This module is compatible with the :mod:`service <salt.states.service>` states,
so it can be used to maintain services using the ``provider`` argument:

.. code-block:: yaml

    myservice:
      service.running:
        - provider: daemontools
"""
import logging
import os
import os.path
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError
__func_alias__ = {'reload_': 'reload'}
log = logging.getLogger(__name__)
__virtualname__ = 'daemontools'
VALID_SERVICE_DIRS = ['/service', '/var/service', '/etc/service']
SERVICE_DIR = None
for service_dir in VALID_SERVICE_DIRS:
    if os.path.exists(service_dir):
        SERVICE_DIR = service_dir
        break

def __virtual__():
    if False:
        while True:
            i = 10
    BINS = frozenset(('svc', 'supervise', 'svok'))
    if all((salt.utils.path.which(b) for b in BINS)) and SERVICE_DIR:
        return __virtualname__
    return (False, 'Missing dependency: {}'.format(BINS))

def _service_path(name):
    if False:
        while True:
            i = 10
    '\n    build service path\n    '
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    return '{}/{}'.format(SERVICE_DIR, name)

def start(name):
    if False:
        while True:
            i = 10
    "\n    Starts service via daemontools\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.start <service name>\n    "
    __salt__['file.remove']('{}/down'.format(_service_path(name)))
    cmd = 'svc -u {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def stop(name):
    if False:
        return 10
    "\n    Stops service via daemontools\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.stop <service name>\n    "
    __salt__['file.touch']('{}/down'.format(_service_path(name)))
    cmd = 'svc -d {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def term(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Send a TERM to service via daemontools\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.term <service name>\n    "
    cmd = 'svc -t {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def reload_(name):
    if False:
        while True:
            i = 10
    "\n    Wrapper for term()\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.reload <service name>\n    "
    term(name)

def restart(name):
    if False:
        i = 10
        return i + 15
    "\n    Restart service via daemontools. This will stop/start service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.restart <service name>\n    "
    ret = 'restart False'
    if stop(name) and start(name):
        ret = 'restart True'
    return ret

def full_restart(name):
    if False:
        print('Hello World!')
    "\n    Calls daemontools.restart() function\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.full_restart <service name>\n    "
    restart(name)

def status(name, sig=None):
    if False:
        return 10
    "\n    Return the status for a service via daemontools, return pid if running\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.status <service name>\n    "
    cmd = 'svstat {}'.format(_service_path(name))
    out = __salt__['cmd.run_stdout'](cmd, python_shell=False)
    try:
        pid = re.search('\\(pid (\\d+)\\)', out).group(1)
    except AttributeError:
        pid = ''
    return pid

def available(name):
    if False:
        while True:
            i = 10
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.available foo\n    "
    return name in get_all()

def missing(name):
    if False:
        print('Hello World!')
    "\n    The inverse of daemontools.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.missing foo\n    "
    return name not in get_all()

def get_all():
    if False:
        print('Hello World!')
    "\n    Return a list of all available services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.get_all\n    "
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    return sorted(os.listdir(SERVICE_DIR))

def enabled(name, **kwargs):
    if False:
        print('Hello World!')
    '\n    Return True if the named service is enabled, false otherwise\n    A service is considered enabled if in your service directory:\n    - an executable ./run file exist\n    - a file named "down" does not exist\n\n    .. versionadded:: 2015.5.7\n\n    name\n        Service name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' daemontools.enabled <service name>\n    '
    if not available(name):
        log.error('Service %s not found', name)
        return False
    run_file = os.path.join(SERVICE_DIR, name, 'run')
    down_file = os.path.join(SERVICE_DIR, name, 'down')
    return os.path.isfile(run_file) and os.access(run_file, os.X_OK) and (not os.path.isfile(down_file))

def disabled(name):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is enabled, false otherwise\n\n    .. versionadded:: 2015.5.6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' daemontools.disabled <service name>\n    "
    return not enabled(name)