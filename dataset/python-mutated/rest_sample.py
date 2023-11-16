"""
Generate baseline proxy minion grains
"""
import salt.utils.platform
__proxyenabled__ = ['rest_sample']
__virtualname__ = 'rest_sample'

def __virtual__():
    if False:
        i = 10
        return i + 15
    try:
        if salt.utils.platform.is_proxy() and __opts__['proxy']['proxytype'] == 'rest_sample':
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
        for i in range(10):
            print('nop')
    '\n    The loader will execute functions with one argument and pass\n    a reference to the proxymodules LazyLoader object.  However,\n    grains sometimes get called before the LazyLoader object is setup\n    so `proxy` might be None.\n    '
    if proxy:
        return {'proxy_functions': proxy['rest_sample.fns']()}

def os():
    if False:
        for i in range(10):
            print('nop')
    return {'os': 'RestExampleOS'}

def location():
    if False:
        while True:
            i = 10
    return {'location': 'In this darn virtual machine.  Let me out!'}

def os_family():
    if False:
        while True:
            i = 10
    return {'os_family': 'proxy'}

def os_data():
    if False:
        return 10
    return {'os_data': 'funkyHttp release 1.0.a.4.g'}