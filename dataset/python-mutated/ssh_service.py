"""
Provide the service module for the proxy-minion SSH sample
.. versionadded:: 2015.8.2
"""
import fnmatch
import logging
import re
import salt.utils.platform
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'service'

def __virtual__():
    if False:
        return 10
    '\n    Only work on systems that are a proxy minion\n    '
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'ssh_sample':
            return __virtualname__
    except KeyError:
        return (False, 'The ssh_service execution module failed to load. Check the proxy key in pillar.')
    return (False, 'The ssh_service execution module failed to load: only works on an ssh_sample proxy minion.')

def get_all():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of all available services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    proxy_fn = 'ssh_sample.service_list'
    return __proxy__[proxy_fn]()

def list_():
    if False:
        return 10
    "\n    Return a list of all available services.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.list\n    "
    return get_all()

def start(name, sig=None):
    if False:
        return 10
    "\n    Start the specified service on the ssh_sample\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    proxy_fn = 'ssh_sample.service_start'
    return __proxy__[proxy_fn](name)

def stop(name, sig=None):
    if False:
        print('Hello World!')
    "\n    Stop the specified service on the rest_sample\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    proxy_fn = 'ssh_sample.service_stop'
    return __proxy__[proxy_fn](name)

def restart(name, sig=None):
    if False:
        return 10
    "\n    Restart the specified service with rest_sample\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    proxy_fn = 'ssh_sample.service_restart'
    return __proxy__[proxy_fn](name)

def status(name, sig=None):
    if False:
        while True:
            i = 10
    "\n    Return the status for a service via ssh_sample.\n    If the name contains globbing, a dict mapping service name to True/False\n    values is returned.\n\n    .. versionchanged:: 2018.3.0\n        The service name can now be a glob (e.g. ``salt*``)\n\n    Args:\n        name (str): The name of the service to check\n        sig (str): Not implemented\n\n    Returns:\n        bool: True if running, False otherwise\n        dict: Maps service name to True if running, False otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name>\n    "
    proxy_fn = 'ssh_sample.service_status'
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        resp = __proxy__[proxy_fn](service)
        if resp['comment'] == 'running':
            results[service] = True
        else:
            results[service] = False
    if contains_globbing:
        return results
    return results[name]

def running(name, sig=None):
    if False:
        while True:
            i = 10
    '\n    Return whether this service is running.\n    '
    return status(name).get(name, False)

def enabled(name, sig=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Only the 'redbull' service is 'enabled' in the test\n    "
    return name == 'redbull'