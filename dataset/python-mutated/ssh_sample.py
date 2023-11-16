"""
Generate baseline proxy minion grains
"""
import salt.utils.platform
__proxyenabled__ = ['ssh_sample']
__virtualname__ = 'ssh_sample'

def __virtual__():
    if False:
        print('Hello World!')
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'ssh_sample':
            return __virtualname__
    except KeyError:
        pass
    return False

def kernel():
    if False:
        i = 10
        return i + 15
    return {'kernel': 'proxy'}

def proxy_functions(proxy):
    if False:
        i = 10
        return i + 15
    '\n    The loader will execute functions with one argument and pass\n    a reference to the proxymodules LazyLoader object.  However,\n    grains sometimes get called before the LazyLoader object is setup\n    so `proxy` might be None.\n    '
    return {'proxy_functions': proxy['ssh_sample.fns']()}

def location():
    if False:
        for i in range(10):
            print('nop')
    return {'location': 'At the other end of an SSH Tunnel!!'}

def os_data():
    if False:
        print('Hello World!')
    return {'os_data': 'DumbShell Endpoint release 4.09.g'}