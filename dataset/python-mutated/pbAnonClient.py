"""
Client which will talk to the server run by pbAnonServer.py, logging in
either anonymously or with username/password credentials.
"""
from sys import stdout
from twisted.cred.credentials import Anonymous, UsernamePassword
from twisted.internet import reactor
from twisted.internet.defer import gatherResults
from twisted.python.log import err, startLogging
from twisted.spread.pb import PBClientFactory

def error(why, msg):
    if False:
        i = 10
        return i + 15
    "\n    Catch-all errback which simply logs the failure.  This isn't expected to\n    be invoked in the normal case for this example.\n    "
    err(why, msg)

def connected(perspective):
    if False:
        return 10
    '\n    Login callback which invokes the remote "foo" method on the perspective\n    which the server returned.\n    '
    print('got perspective1 ref:', perspective)
    print('asking it to foo(13)')
    return perspective.callRemote('foo', 13)

def finished(ignored):
    if False:
        return 10
    '\n    Callback invoked when both logins and method calls have finished to shut\n    down the reactor so the example exits.\n    '
    reactor.stop()

def main():
    if False:
        print('Hello World!')
    '\n    Connect to a PB server running on port 8800 on localhost and log in to\n    it, both anonymously and using a username/password it will recognize.\n    '
    startLogging(stdout)
    factory = PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    anonymousLogin = factory.login(Anonymous())
    anonymousLogin.addCallback(connected)
    anonymousLogin.addErrback(error, 'Anonymous login failed')
    usernameLogin = factory.login(UsernamePassword('user1', 'pass1'))
    usernameLogin.addCallback(connected)
    usernameLogin.addErrback(error, 'Username/password login failed')
    bothDeferreds = gatherResults([anonymousLogin, usernameLogin])
    bothDeferreds.addCallback(finished)
    reactor.run()
if __name__ == '__main__':
    main()