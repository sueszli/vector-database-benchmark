"""
Execution module for Cisco Network Services Orchestrator Proxy minions

.. versionadded:: 2016.11.0

For documentation on setting up the cisconso proxy minion look in the documentation
for :mod:`salt.proxy.cisconso<salt.proxy.cisconso>`.
"""
import salt.utils.platform
__proxyenabled__ = ['cisconso']
__virtualname__ = 'cisconso'

def __virtual__():
    if False:
        print('Hello World!')
    if salt.utils.platform.is_proxy():
        return __virtualname__
    return (False, 'The cisconso execution module failed to load: only available on proxy minions.')

def info():
    if False:
        i = 10
        return i + 15
    "\n    Return system information for grains of the NSO proxy minion\n\n    .. code-block:: bash\n\n        salt '*' cisconso.info\n    "
    return _proxy_cmd('info')

def get_data(datastore, path):
    if False:
        while True:
            i = 10
    "\n    Get the configuration of the device tree at the given path\n\n    :param datastore: The datastore, e.g. running, operational.\n        One of the NETCONF store IETF types\n    :type  datastore: :class:`DatastoreType` (``str`` enum).\n\n    :param path: The device path to set the value at,\n        a list of element names in order, / separated\n    :type  path: ``list``, ``str`` OR ``tuple``\n\n    :return: The network configuration at that tree\n    :rtype: ``dict``\n\n    .. code-block:: bash\n\n        salt cisco-nso cisconso.get_data running 'devices/ex0'\n    "
    if isinstance(path, str):
        path = '/'.split(path)
    return _proxy_cmd('get_data', datastore, path)

def set_data_value(datastore, path, data):
    if False:
        print('Hello World!')
    "\n    Set a data entry in a datastore\n\n    :param datastore: The datastore, e.g. running, operational.\n        One of the NETCONF store IETF types\n    :type  datastore: :class:`DatastoreType` (``str`` enum).\n\n    :param path: The device path to set the value at,\n        a list of element names in order, / separated\n    :type  path: ``list``, ``str`` OR ``tuple``\n\n    :param data: The new value at the given path\n    :type  data: ``dict``\n\n    :rtype: ``bool``\n    :return: ``True`` if successful, otherwise error.\n\n    .. code-block:: bash\n\n        salt cisco-nso cisconso.set_data_value running 'devices/ex0/routes' 10.0.0.20/24\n    "
    if isinstance(path, str):
        path = '/'.split(path)
    return _proxy_cmd('set_data_value', datastore, path, data)

def get_rollbacks():
    if False:
        return 10
    '\n    Get a list of stored configuration rollbacks\n\n    .. code-block:: bash\n\n        salt cisco-nso cisconso.get_rollbacks\n    '
    return _proxy_cmd('get_rollbacks')

def get_rollback(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the backup of stored a configuration rollback\n\n    :param name: Typically an ID of the backup\n    :type  name: ``str``\n\n    :rtype: ``str``\n    :return: the contents of the rollback snapshot\n\n    .. code-block:: bash\n\n        salt cisco-nso cisconso.get_rollback 52\n    '
    return _proxy_cmd('get_rollback', name)

def apply_rollback(datastore, name):
    if False:
        while True:
            i = 10
    '\n    Apply a system rollback\n\n    :param datastore: The datastore, e.g. running, operational.\n        One of the NETCONF store IETF types\n    :type  datastore: :class:`DatastoreType` (``str`` enum).\n\n    :param name: an ID of the rollback to restore\n    :type  name: ``str``\n\n    .. code-block:: bash\n\n        salt cisco-nso cisconso.apply_rollback 52\n    '
    return _proxy_cmd('apply_rollback', datastore, name)

def _proxy_cmd(command, *args, **kwargs):
    if False:
        print('Hello World!')
    '\n    run commands from __proxy__\n    :mod:`salt.proxy.cisconso<salt.proxy.cisconso>`\n\n    command\n        function from `salt.proxy.cisconso` to run\n\n    args\n        positional args to pass to `command` function\n\n    kwargs\n        key word arguments to pass to `command` function\n    '
    proxy_prefix = __opts__['proxy']['proxytype']
    proxy_cmd = '.'.join([proxy_prefix, command])
    if proxy_cmd not in __proxy__:
        return False
    for k in kwargs:
        if k.startswith('__pub_'):
            kwargs.pop(k)
    return __proxy__[proxy_cmd](*args, **kwargs)