from sys import stdout
from twisted.python import log
log.discardLogs()
from twisted.internet import reactor
from twisted.spread import pb

def connected(root):
    if False:
        print('Hello World!')
    root.callRemote('nextQuote').addCallbacks(success, failure)

def success(quote):
    if False:
        print('Hello World!')
    stdout.write(quote + '\n')
    reactor.stop()

def failure(error):
    if False:
        while True:
            i = 10
    stdout.write('Failed to obtain quote.\n')
    reactor.stop()
factory = pb.PBClientFactory()
reactor.connectTCP('localhost', pb.portno, factory)
factory.getRootObject().addCallbacks(connected, failure)
reactor.run()