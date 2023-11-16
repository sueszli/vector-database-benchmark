"""
Service support for the REST example
"""
import logging
import salt.utils.platform
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        return 10
    '\n    Only work on proxy\n    '
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'ssh_sample':
            return __virtualname__
    except KeyError:
        return (False, 'The ssh_package execution module failed to load. Check the proxy key in pillar.')
    return (False, 'The ssh_package execution module failed to load: only works on an ssh_sample proxy minion.')

def list_pkgs(versions_as_list=False, **kwargs):
    if False:
        print('Hello World!')
    return __proxy__['ssh_sample.package_list']()

def install(name=None, refresh=False, fromrepo=None, pkgs=None, sources=None, **kwargs):
    if False:
        i = 10
        return i + 15
    return __proxy__['ssh_sample.package_install'](name, **kwargs)

def remove(name=None, pkgs=None, **kwargs):
    if False:
        return 10
    return __proxy__['ssh_sample.package_remove'](name)