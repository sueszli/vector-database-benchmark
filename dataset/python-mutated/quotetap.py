from TwistedQuotes import quoteproto
from TwistedQuotes import quoters
from twisted.application import internet
from twisted.python import usage

class Options(usage.Options):
    optParameters = [['port', 'p', 8007, 'Port number to listen on for QOTD protocol.'], ['static', 's', 'An apple a day keeps the doctor away.', 'A static quote to display.'], ['file', 'f', None, 'A fortune-format text file to read quotes from.']]

def makeService(config):
    if False:
        return 10
    'Return a service that will be attached to the application.'
    if config['file']:
        quoter = quoters.FortuneQuoter([config['file']])
    else:
        quoter = quoters.StaticQuoter(config['static'])
    port = int(config['port'])
    factory = quoteproto.QOTDFactory(quoter)
    return internet.TCPServer(port, factory)