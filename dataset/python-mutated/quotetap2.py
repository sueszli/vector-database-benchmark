from TwistedQuotes import pbquote
from TwistedQuotes import quoteproto
from TwistedQuotes import quoters
from twisted.application import internet, service
from twisted.python import usage
from twisted.spread import pb

class Options(usage.Options):
    optParameters = [['port', 'p', 8007, 'Port number to listen on for QOTD protocol.'], ['static', 's', 'An apple a day keeps the doctor away.', 'A static quote to display.'], ['file', 'f', None, 'A fortune-format text file to read quotes from.'], ['pb', 'b', None, 'Port to listen with PB server']]

def makeService(config):
    if False:
        while True:
            i = 10
    svc = service.MultiService()
    if config['file']:
        quoter = quoters.FortuneQuoter([config['file']])
    else:
        quoter = quoters.StaticQuoter(config['static'])
    port = int(config['port'])
    factory = quoteproto.QOTDFactory(quoter)
    pbport = config['pb']
    if pbport:
        pbfact = pb.PBServerFactory(pbquote.QuoteReader(quoter))
        svc.addService(internet.TCPServer(int(pbport), pbfact))
    svc.addService(internet.TCPServer(port, factory))
    return svc