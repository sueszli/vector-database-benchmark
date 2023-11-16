"""
Provide the service module for the dummy proxy used in integration tests
"""
import logging
import salt.utils.platform
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'service'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on systems that are a proxy minion\n    '
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'dummy':
            return __virtualname__
    except KeyError:
        return (False, 'The dummyproxy_service execution module failed to load. Check the proxy key in pillar or /etc/salt/proxy.')
    return (False, 'The dummyproxy_service execution module failed to load: only works on the integration testsuite dummy proxy minion.')

def get_all():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of all available services\n\n    .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    proxy_fn = 'dummy.service_list'
    return __proxy__[proxy_fn]()

def list_():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of all available services.\n\n    .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.list\n    "
    return get_all()

def start(name, sig=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start the specified service on the dummy\n\n    .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    proxy_fn = 'dummy.service_start'
    return __proxy__[proxy_fn](name)

def stop(name, sig=None):
    if False:
        i = 10
        return i + 15
    "\n    Stop the specified service on the dummy\n\n    .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    proxy_fn = 'dummy.service_stop'
    return __proxy__[proxy_fn](name)

def restart(name, sig=None):
    if False:
        print('Hello World!')
    "\n    Restart the specified service with dummy.\n\n    .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    proxy_fn = 'dummy.service_restart'
    return __proxy__[proxy_fn](name)

def status(name, sig=None):
    if False:
        print('Hello World!')
    "\n    Return the status for a service via dummy, returns a bool\n    whether the service is running.\n\n    .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name>\n    "
    proxy_fn = 'dummy.service_status'
    resp = __proxy__[proxy_fn](name)
    if resp['comment'] == 'stopped':
        return False
    if resp['comment'] == 'running':
        return True

def running(name, sig=None):
    if False:
        i = 10
        return i + 15
    '\n    Return whether this service is running.\n\n    .. versionadded:: 2016.11.3\n\n    '
    return status(name).get(name, False)

def enabled(name, sig=None):
    if False:
        while True:
            i = 10
    "\n    Only the 'redbull' service is 'enabled' in the test\n\n    .. versionadded:: 2016.11.3\n\n    "
    return name == 'redbull'