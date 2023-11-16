"""
s6 service module

This module is compatible with the :mod:`service <salt.states.service>` states,
so it can be used to maintain services using the ``provider`` argument:

.. code-block:: yaml

    myservice:
      service:
        - running
        - provider: s6

Note that the ``enabled`` argument is not available with this provider.

:codeauthor: Marek Skrobacki <skrobul@skrobul.com>
"""
import os
import re
from salt.exceptions import CommandExecutionError
__func_alias__ = {'reload_': 'reload'}
VALID_SERVICE_DIRS = ['/service', '/etc/service']
SERVICE_DIR = None
for service_dir in VALID_SERVICE_DIRS:
    if os.path.exists(service_dir):
        SERVICE_DIR = service_dir
        break

def _service_path(name):
    if False:
        i = 10
        return i + 15
    '\n    build service path\n    '
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    return '{}/{}'.format(SERVICE_DIR, name)

def start(name):
    if False:
        return 10
    "\n    Starts service via s6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.start <service name>\n    "
    cmd = 's6-svc -u {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        while True:
            i = 10
    "\n    Stops service via s6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.stop <service name>\n    "
    cmd = 's6-svc -d {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd)

def term(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Send a TERM to service via s6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.term <service name>\n    "
    cmd = 's6-svc -t {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd)

def reload_(name):
    if False:
        while True:
            i = 10
    "\n    Send a HUP to service via s6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.reload <service name>\n    "
    cmd = 's6-svc -h {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        while True:
            i = 10
    "\n    Restart service via s6. This will stop/start service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.restart <service name>\n    "
    cmd = 's6-svc -t {}'.format(_service_path(name))
    return not __salt__['cmd.retcode'](cmd)

def full_restart(name):
    if False:
        i = 10
        return i + 15
    "\n    Calls s6.restart() function\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.full_restart <service name>\n    "
    restart(name)

def status(name, sig=None):
    if False:
        return 10
    "\n    Return the status for a service via s6, return pid if running\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.status <service name>\n    "
    cmd = 's6-svstat {}'.format(_service_path(name))
    out = __salt__['cmd.run_stdout'](cmd)
    try:
        pid = re.search('up \\(pid (\\d+)\\)', out).group(1)
    except AttributeError:
        pid = ''
    return pid

def available(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.available foo\n    "
    return name in get_all()

def missing(name):
    if False:
        i = 10
        return i + 15
    "\n    The inverse of s6.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.missing foo\n    "
    return name not in get_all()

def get_all():
    if False:
        print('Hello World!')
    "\n    Return a list of all available services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' s6.get_all\n    "
    if not SERVICE_DIR:
        raise CommandExecutionError('Could not find service directory.')
    service_list = [dirname for dirname in os.listdir(SERVICE_DIR) if not dirname.startswith('.')]
    return sorted(service_list)