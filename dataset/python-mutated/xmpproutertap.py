from twisted.application import strports
from twisted.python import usage
from twisted.words.protocols.jabber import component

class Options(usage.Options):
    optParameters = [('port', None, 'tcp:5347:interface=127.0.0.1', 'Port components connect to'), ('secret', None, 'secret', 'Router secret')]
    optFlags = [('verbose', 'v', 'Log traffic')]

def makeService(config):
    if False:
        i = 10
        return i + 15
    router = component.Router()
    factory = component.XMPPComponentServerFactory(router, config['secret'])
    if config['verbose']:
        factory.logTraffic = True
    return strports.service(config['port'], factory)