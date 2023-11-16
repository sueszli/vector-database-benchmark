"""
Philips HUE lamps module for proxy.

.. versionadded:: 2015.8.3
"""
import sys
__virtualname__ = 'hue'
__proxyenabled__ = ['philips_hue']

def _proxy():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get proxy.\n    '
    return __proxy__

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Start the Philips HUE only for proxies.\n    '
    if not _proxy():
        return False

    def _mkf(cmd_name, doc):
        if False:
            while True:
                i = 10
        '\n        Nested function to help move proxy functions into sys.modules\n        '

        def _cmd(*args, **kw):
            if False:
                while True:
                    i = 10
            '\n            Call commands in proxy\n            '
            proxyfn = 'philips_hue.' + cmd_name
            return __proxy__[proxyfn](*args, **kw)
        return _cmd
    import salt.proxy.philips_hue as hue
    for method in dir(hue):
        if method.startswith('call_'):
            setattr(sys.modules[__name__], method[5:], _mkf(method, getattr(hue, method).__doc__))
    del hue
    return _proxy() and __virtualname__ or False