""" A temporary placeholder for client-capable strports, until we
sufficient use cases get identified """
from twisted.internet.endpoints import _parse

def _parseTCPSSL(factory, domain, port):
    if False:
        for i in range(10):
            print('nop')
    'For the moment, parse TCP or SSL connections the same'
    return ((domain, int(port), factory), {})

def _parseUNIX(factory, address):
    if False:
        while True:
            i = 10
    return ((address, factory), {})
_funcs = {'tcp': _parseTCPSSL, 'unix': _parseUNIX, 'ssl': _parseTCPSSL}

def parse(description, factory):
    if False:
        print('Hello World!')
    (args, kw) = _parse(description)
    return (args[0].upper(),) + _funcs[args[0]](factory, *args[1:], **kw)

def client(description, factory):
    if False:
        return 10
    from twisted.application import internet
    (name, args, kw) = parse(description, factory)
    return getattr(internet, name + 'Client')(*args, **kw)