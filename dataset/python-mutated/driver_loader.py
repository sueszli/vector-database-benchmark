"""
Module containing utility functions for loading adapter / driver class instances for drivers using
stevedore dynamic plugin loading.
"""
from st2common import log as logging
__all__ = ['get_available_backends', 'get_backend_driver', 'get_backend_instance']
LOG = logging.getLogger(__name__)
BACKENDS_NAMESPACE = 'st2common.rbac.backend'

def get_available_backends(namespace, invoke_on_load=False):
    if False:
        print('Hello World!')
    '\n    Return names of the available / installed backends.\n\n    :rtype: ``list`` of ``str``\n    '
    from stevedore.extension import ExtensionManager
    manager = ExtensionManager(namespace=namespace, invoke_on_load=invoke_on_load)
    return manager.names()

def get_backend_driver(namespace, name, invoke_on_load=False):
    if False:
        return 10
    '\n    Retrieve a driver (module / class / function) the provided backend.\n\n    :param name: Backend name.\n    :type name: ``str``\n    '
    from stevedore.driver import DriverManager
    LOG.debug('Retrieving driver for backend "%s"' % name)
    try:
        manager = DriverManager(namespace=namespace, name=name, invoke_on_load=invoke_on_load)
    except RuntimeError:
        message = 'Invalid "%s" backend specified: %s' % (namespace, name)
        LOG.exception(message)
        raise ValueError(message)
    return manager.driver

def get_backend_instance(namespace, name, invoke_on_load=False):
    if False:
        i = 10
        return i + 15
    '\n    Retrieve a class instance for the provided backend.\n\n    :param name: Backend name.\n    :type name: ``str``\n    '
    cls = get_backend_driver(namespace=namespace, name=name, invoke_on_load=invoke_on_load)
    cls_instance = cls()
    return cls_instance