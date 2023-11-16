"""
L{twisted.words} support for Instance Messenger.
"""
from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE

class TwistedWordsPerson(basesupport.AbstractPerson):
    """I a facade for a person you can talk to through a twisted.words service."""

    def __init__(self, name, wordsAccount):
        if False:
            for i in range(10):
                print('nop')
        basesupport.AbstractPerson.__init__(self, name, wordsAccount)
        self.status = OFFLINE

    def isOnline(self):
        if False:
            i = 10
            return i + 15
        return self.status == ONLINE or self.status == AWAY

    def getStatus(self):
        if False:
            i = 10
            return i + 15
        return self.status

    def sendMessage(self, text, metadata):
        if False:
            print('Hello World!')
        'Return a deferred...'
        if metadata:
            d = self.account.client.perspective.directMessage(self.name, text, metadata)
            d.addErrback(self.metadataFailed, '* ' + text)
            return d
        else:
            return self.account.client.perspective.callRemote('directMessage', self.name, text)

    def metadataFailed(self, result, text):
        if False:
            return 10
        print('result:', result, 'text:', text)
        return self.account.client.perspective.directMessage(self.name, text)

    def setStatus(self, status):
        if False:
            print('Hello World!')
        self.status = status
        self.chat.getContactsList().setContactStatus(self)

@implementer(interfaces.IGroup)
class TwistedWordsGroup(basesupport.AbstractGroup):

    def __init__(self, name, wordsClient):
        if False:
            i = 10
            return i + 15
        basesupport.AbstractGroup.__init__(self, name, wordsClient)
        self.joined = 0

    def sendGroupMessage(self, text, metadata=None):
        if False:
            while True:
                i = 10
        'Return a deferred.'
        if metadata:
            d = self.account.client.perspective.callRemote('groupMessage', self.name, text, metadata)
            d.addErrback(self.metadataFailed, '* ' + text)
            return d
        else:
            return self.account.client.perspective.callRemote('groupMessage', self.name, text)

    def setTopic(self, text):
        if False:
            i = 10
            return i + 15
        self.account.client.perspective.callRemote('setGroupMetadata', {'topic': text, 'topic_author': self.client.name}, self.name)

    def metadataFailed(self, result, text):
        if False:
            return 10
        print('result:', result, 'text:', text)
        return self.account.client.perspective.callRemote('groupMessage', self.name, text)

    def joining(self):
        if False:
            return 10
        self.joined = 1

    def leaving(self):
        if False:
            print('Hello World!')
        self.joined = 0

    def leave(self):
        if False:
            return 10
        return self.account.client.perspective.callRemote('leaveGroup', self.name)

class TwistedWordsClient(pb.Referenceable, basesupport.AbstractClientMixin):
    """In some cases, this acts as an Account, since it a source of text
    messages (multiple Words instances may be on a single PB connection)
    """

    def __init__(self, acct, serviceName, perspectiveName, chatui, _logonDeferred=None):
        if False:
            print('Hello World!')
        self.accountName = '{} ({}:{})'.format(acct.accountName, serviceName, perspectiveName)
        self.name = perspectiveName
        print('HELLO I AM A PB SERVICE', serviceName, perspectiveName)
        self.chat = chatui
        self.account = acct
        self._logonDeferred = _logonDeferred

    def getPerson(self, name):
        if False:
            return 10
        return self.chat.getPerson(name, self)

    def getGroup(self, name):
        if False:
            while True:
                i = 10
        return self.chat.getGroup(name, self)

    def getGroupConversation(self, name):
        if False:
            while True:
                i = 10
        return self.chat.getGroupConversation(self.getGroup(name))

    def addContact(self, name):
        if False:
            return 10
        self.perspective.callRemote('addContact', name)

    def remote_receiveGroupMembers(self, names, group):
        if False:
            for i in range(10):
                print('nop')
        print('received group members:', names, group)
        self.getGroupConversation(group).setGroupMembers(names)

    def remote_receiveGroupMessage(self, sender, group, message, metadata=None):
        if False:
            print('Hello World!')
        print('received a group message', sender, group, message, metadata)
        self.getGroupConversation(group).showGroupMessage(sender, message, metadata)

    def remote_memberJoined(self, member, group):
        if False:
            print('Hello World!')
        print('member joined', member, group)
        self.getGroupConversation(group).memberJoined(member)

    def remote_memberLeft(self, member, group):
        if False:
            i = 10
            return i + 15
        print('member left')
        self.getGroupConversation(group).memberLeft(member)

    def remote_notifyStatusChanged(self, name, status):
        if False:
            print('Hello World!')
        self.chat.getPerson(name, self).setStatus(status)

    def remote_receiveDirectMessage(self, name, message, metadata=None):
        if False:
            i = 10
            return i + 15
        self.chat.getConversation(self.chat.getPerson(name, self)).showMessage(message, metadata)

    def remote_receiveContactList(self, clist):
        if False:
            return 10
        for (name, status) in clist:
            self.chat.getPerson(name, self).setStatus(status)

    def remote_setGroupMetadata(self, dict_, groupName):
        if False:
            print('Hello World!')
        if 'topic' in dict_:
            self.getGroupConversation(groupName).setTopic(dict_['topic'], dict_.get('topic_author', None))

    def joinGroup(self, name):
        if False:
            while True:
                i = 10
        self.getGroup(name).joining()
        return self.perspective.callRemote('joinGroup', name).addCallback(self._cbGroupJoined, name)

    def leaveGroup(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.getGroup(name).leaving()
        return self.perspective.callRemote('leaveGroup', name).addCallback(self._cbGroupLeft, name)

    def _cbGroupJoined(self, result, name):
        if False:
            for i in range(10):
                print('nop')
        groupConv = self.chat.getGroupConversation(self.getGroup(name))
        groupConv.showGroupMessage('sys', 'you joined')
        self.perspective.callRemote('getGroupMembers', name)

    def _cbGroupLeft(self, result, name):
        if False:
            return 10
        print('left', name)
        groupConv = self.chat.getGroupConversation(self.getGroup(name), 1)
        groupConv.showGroupMessage('sys', 'you left')

    def connected(self, perspective):
        if False:
            i = 10
            return i + 15
        print('Connected Words Client!', perspective)
        if self._logonDeferred is not None:
            self._logonDeferred.callback(self)
        self.perspective = perspective
        self.chat.getContactsList()
pbFrontEnds = {'twisted.words': TwistedWordsClient, 'twisted.reality': None}

@implementer(interfaces.IAccount)
class PBAccount(basesupport.AbstractAccount):
    gatewayType = 'PB'
    _groupFactory = TwistedWordsGroup
    _personFactory = TwistedWordsPerson

    def __init__(self, accountName, autoLogin, username, password, host, port, services=None):
        if False:
            return 10
        '\n        @param username: The name of your PB Identity.\n        @type username: string\n        '
        basesupport.AbstractAccount.__init__(self, accountName, autoLogin, username, password, host, port)
        self.services = []
        if not services:
            services = [('twisted.words', 'twisted.words', username)]
        for (serviceType, serviceName, perspectiveName) in services:
            self.services.append([pbFrontEnds[serviceType], serviceName, perspectiveName])

    def logOn(self, chatui):
        if False:
            i = 10
            return i + 15
        '\n        @returns: this breaks with L{interfaces.IAccount}\n        @returntype: DeferredList of L{interfaces.IClient}s\n        '
        if not self._isConnecting and (not self._isOnline):
            self._isConnecting = 1
            d = self._startLogOn(chatui)
            d.addErrback(self._loginFailed)

            def registerMany(results):
                if False:
                    return 10
                for (success, result) in results:
                    if success:
                        chatui.registerAccountClient(result)
                        self._cb_logOn(result)
                    else:
                        log.err(result)
            d.addCallback(registerMany)
            return d
        else:
            raise error.ConnectionError('Connection in progress')

    def logOff(self):
        if False:
            i = 10
            return i + 15
        pass

    def _startLogOn(self, chatui):
        if False:
            i = 10
            return i + 15
        print('Connecting...', end=' ')
        d = pb.getObjectAt(self.host, self.port)
        d.addCallbacks(self._cbConnected, self._ebConnected, callbackArgs=(chatui,))
        return d

    def _cbConnected(self, root, chatui):
        if False:
            print('Hello World!')
        print('Connected!')
        print('Identifying...', end=' ')
        d = pb.authIdentity(root, self.username, self.password)
        d.addCallbacks(self._cbIdent, self._ebConnected, callbackArgs=(chatui,))
        return d

    def _cbIdent(self, ident, chatui):
        if False:
            for i in range(10):
                print('nop')
        if not ident:
            print('falsely identified.')
            return self._ebConnected(Failure(Exception('username or password incorrect')))
        print('Identified!')
        dl = []
        for (handlerClass, sname, pname) in self.services:
            d = defer.Deferred()
            dl.append(d)
            handler = handlerClass(self, sname, pname, chatui, d)
            ident.callRemote('attach', sname, pname, handler).addCallback(handler.connected)
        return defer.DeferredList(dl)

    def _ebConnected(self, error):
        if False:
            i = 10
            return i + 15
        print('Not connected.')
        return error