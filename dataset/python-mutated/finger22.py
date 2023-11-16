from zope.interface import Interface, implementer
from OpenSSL import SSL
from twisted.application import internet, service, strports
from twisted.internet import defer, endpoints, protocol, reactor
from twisted.protocols import basic
from twisted.python import components
from twisted.spread import pb
from twisted.web import resource, server, static, xmlrpc
from twisted.words.protocols import irc

class IFingerService(Interface):

    def getUser(user):
        if False:
            print('Hello World!')
        '\n        Return a deferred returning L{bytes}.\n        '

    def getUsers():
        if False:
            i = 10
            return i + 15
        '\n        Return a deferred returning a L{list} of L{bytes}.\n        '

class IFingerSetterService(Interface):

    def setUser(user, status):
        if False:
            i = 10
            return i + 15
        "\n        Set the user's status to something.\n        "

def catchError(err):
    if False:
        for i in range(10):
            print('nop')
    return 'Internal error in server'

class FingerProtocol(basic.LineReceiver):

    def lineReceived(self, user):
        if False:
            return 10
        d = self.factory.getUser(user)
        d.addErrback(catchError)

        def writeValue(value):
            if False:
                return 10
            self.transport.write(value + b'\r\n')
            self.transport.loseConnection()
        d.addCallback(writeValue)

class IFingerFactory(Interface):

    def getUser(user):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a deferred returning a string.\n        '

    def buildProtocol(addr):
        if False:
            return 10
        '\n        Return a protocol returning a string.\n        '

@implementer(IFingerFactory)
class FingerFactoryFromService(protocol.ServerFactory):
    protocol = FingerProtocol

    def __init__(self, service):
        if False:
            for i in range(10):
                print('nop')
        self.service = service

    def getUser(self, user):
        if False:
            return 10
        return self.service.getUser(user)
components.registerAdapter(FingerFactoryFromService, IFingerService, IFingerFactory)

class FingerSetterProtocol(basic.LineReceiver):

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.lines = []

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        self.lines.append(line)

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        if len(self.lines) == 2:
            self.factory.setUser(*self.lines)

class IFingerSetterFactory(Interface):

    def setUser(user, status):
        if False:
            return 10
        '\n        Return a deferred returning L{bytes}.\n        '

    def buildProtocol(addr):
        if False:
            i = 10
            return i + 15
        '\n        Return a protocol returning L{bytes}.\n        '

@implementer(IFingerSetterFactory)
class FingerSetterFactoryFromService(protocol.ServerFactory):
    protocol = FingerSetterProtocol

    def __init__(self, service):
        if False:
            while True:
                i = 10
        self.service = service

    def setUser(self, user, status):
        if False:
            i = 10
            return i + 15
        self.service.setUser(user, status)
components.registerAdapter(FingerSetterFactoryFromService, IFingerSetterService, IFingerSetterFactory)

class IRCReplyBot(irc.IRCClient):

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.nickname = self.factory.nickname
        irc.IRCClient.connectionMade(self)

    def privmsg(self, user, channel, msg):
        if False:
            while True:
                i = 10
        user = user.split('!')[0]
        if self.nickname.lower() == channel.lower():
            d = self.factory.getUser(msg.encode('ascii'))
            d.addErrback(catchError)
            d.addCallback(lambda m: f'Status of {msg}: {m}')
            d.addCallback(lambda m: self.msg(user, m))

class IIRCClientFactory(Interface):
    """
    @ivar nickname
    """

    def getUser(user):
        if False:
            while True:
                i = 10
        '\n        Return a deferred returning a string.\n        '

    def buildProtocol(addr):
        if False:
            print('Hello World!')
        '\n        Return a protocol.\n        '

@implementer(IIRCClientFactory)
class IRCClientFactoryFromService(protocol.ClientFactory):
    protocol = IRCReplyBot
    nickname = None

    def __init__(self, service):
        if False:
            return 10
        self.service = service

    def getUser(self, user):
        if False:
            return 10
        return self.service.getUser(user)
components.registerAdapter(IRCClientFactoryFromService, IFingerService, IIRCClientFactory)

class UserStatusTree(resource.Resource):

    def __init__(self, service):
        if False:
            for i in range(10):
                print('nop')
        resource.Resource.__init__(self)
        self.service = service
        self.putChild('RPC2', UserStatusXR(self.service))
        self.putChild('', self)

    def _cb_render_GET(self, users, request):
        if False:
            for i in range(10):
                print('nop')
        userOutput = ''.join([f'<li><a href="{user}">{user}</a></li>' for user in users])
        request.write('\n            <html><head><title>Users</title></head><body>\n            <h1>Users</h1>\n            <ul>\n            %s\n            </ul></body></html>' % userOutput)
        request.finish()

    def render_GET(self, request):
        if False:
            return 10
        d = self.service.getUsers()
        d.addCallback(self._cb_render_GET, request)
        return server.NOT_DONE_YET

    def getChild(self, path, request):
        if False:
            print('Hello World!')
        return UserStatus(user=path, service=self.service)
components.registerAdapter(UserStatusTree, IFingerService, resource.IResource)

class UserStatus(resource.Resource):

    def __init__(self, user, service):
        if False:
            while True:
                i = 10
        resource.Resource.__init__(self)
        self.user = user
        self.service = service

    def _cb_render_GET(self, status, request):
        if False:
            while True:
                i = 10
        request.write('<html><head><title>%s</title></head>\n        <body><h1>%s</h1>\n        <p>%s</p>\n        </body></html>' % (self.user, self.user, status))
        request.finish()

    def render_GET(self, request):
        if False:
            for i in range(10):
                print('nop')
        d = self.service.getUser(self.user)
        d.addCallback(self._cb_render_GET, request)
        return server.NOT_DONE_YET

class UserStatusXR(xmlrpc.XMLRPC):

    def __init__(self, service):
        if False:
            while True:
                i = 10
        xmlrpc.XMLRPC.__init__(self)
        self.service = service

    def xmlrpc_getUser(self, user):
        if False:
            for i in range(10):
                print('nop')
        return self.service.getUser(user)

    def xmlrpc_getUsers(self):
        if False:
            for i in range(10):
                print('nop')
        return self.service.getUsers()

class IPerspectiveFinger(Interface):

    def remote_getUser(username):
        if False:
            print('Hello World!')
        "\n        Return a user's status.\n        "

    def remote_getUsers():
        if False:
            print('Hello World!')
        "\n        Return a user's status.\n        "

@implementer(IPerspectiveFinger)
class PerspectiveFingerFromService(pb.Root):

    def __init__(self, service):
        if False:
            for i in range(10):
                print('nop')
        self.service = service

    def remote_getUser(self, username):
        if False:
            return 10
        return self.service.getUser(username)

    def remote_getUsers(self):
        if False:
            while True:
                i = 10
        return self.service.getUsers()
components.registerAdapter(PerspectiveFingerFromService, IFingerService, IPerspectiveFinger)

@implementer(IFingerService)
class FingerService(service.Service):

    def __init__(self, filename):
        if False:
            while True:
                i = 10
        self.filename = filename
        self.users = {}

    def _read(self):
        if False:
            return 10
        self.users.clear()
        with open(self.filename, 'rb') as f:
            for line in f:
                (user, status) = line.split(b':', 1)
                user = user.strip()
                status = status.strip()
                self.users[user] = status
        self.call = reactor.callLater(30, self._read)

    def getUser(self, user):
        if False:
            while True:
                i = 10
        return defer.succeed(self.users.get(user, b'No such user'))

    def getUsers(self):
        if False:
            while True:
                i = 10
        return defer.succeed(list(self.users.keys()))

    def startService(self):
        if False:
            while True:
                i = 10
        self._read()
        service.Service.startService(self)

    def stopService(self):
        if False:
            print('Hello World!')
        service.Service.stopService(self)
        self.call.cancel()
application = service.Application('finger', uid=1, gid=1)
f = FingerService('/etc/users')
serviceCollection = service.IServiceCollection(application)
f.setServiceParent(serviceCollection)
strports.service('tcp:79', IFingerFactory(f)).setServiceParent(serviceCollection)
site = server.Site(resource.IResource(f))
strports.service('tcp:8000', site).setServiceParent(serviceCollection)
strports.service('ssl:port=443:certKey=cert.pem:privateKey=key.pem', site).setServiceParent(serviceCollection)
i = IIRCClientFactory(f)
i.nickname = 'fingerbot'
internet.ClientService(endpoints.clientFromString(reactor, 'tcp:irc.freenode.org:6667'), i).setServiceParent(serviceCollection)
strports.service('tcp:8889', pb.PBServerFactory(IPerspectiveFinger(f))).setServiceParent(serviceCollection)