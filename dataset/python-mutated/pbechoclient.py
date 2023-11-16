from pbecho import DefinedError
from twisted.cred.credentials import UsernamePassword
from twisted.internet import reactor
from twisted.spread import pb

def success(message):
    if False:
        i = 10
        return i + 15
    print('Message received:', message)

def failure(error):
    if False:
        print('Hello World!')
    t = error.trap(DefinedError)
    print('error received:', t)
    reactor.stop()

def connected(perspective):
    if False:
        print('Hello World!')
    perspective.callRemote('echo', 'hello world').addCallbacks(success, failure)
    perspective.callRemote('error').addCallbacks(success, failure)
    print('connected.')
factory = pb.PBClientFactory()
reactor.connectTCP('localhost', pb.portno, factory)
factory.login(UsernamePassword('guest', 'guest')).addCallbacks(connected, failure)
reactor.run()