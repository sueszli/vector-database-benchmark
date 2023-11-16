"""
IRC support for Instance Messenger.
"""
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc

class IRCPerson(basesupport.AbstractPerson):

    def imperson_whois(self):
        if False:
            print('Hello World!')
        if self.account.client is None:
            raise locals.OfflineError
        self.account.client.sendLine('WHOIS %s' % self.name)

    def isOnline(self):
        if False:
            for i in range(10):
                print('nop')
        return ONLINE

    def getStatus(self):
        if False:
            for i in range(10):
                print('nop')
        return ONLINE

    def setStatus(self, status):
        if False:
            i = 10
            return i + 15
        self.status = status
        self.chat.getContactsList().setContactStatus(self)

    def sendMessage(self, text, meta=None):
        if False:
            for i in range(10):
                print('nop')
        if self.account.client is None:
            raise locals.OfflineError
        for line in text.split('\n'):
            if meta and meta.get('style', None) == 'emote':
                self.account.client.ctcpMakeQuery(self.name, [('ACTION', line)])
            else:
                self.account.client.msg(self.name, line)
        return succeed(text)

@implementer(interfaces.IGroup)
class IRCGroup(basesupport.AbstractGroup):

    def imgroup_testAction(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def imtarget_kick(self, target):
        if False:
            i = 10
            return i + 15
        if self.account.client is None:
            raise locals.OfflineError
        reason = 'for great justice!'
        self.account.client.sendLine(f'KICK #{self.name} {target.name} :{reason}')

    def setTopic(self, topic):
        if False:
            while True:
                i = 10
        if self.account.client is None:
            raise locals.OfflineError
        self.account.client.topic(self.name, topic)

    def sendGroupMessage(self, text, meta={}):
        if False:
            print('Hello World!')
        if self.account.client is None:
            raise locals.OfflineError
        if meta and meta.get('style', None) == 'emote':
            self.account.client.ctcpMakeQuery(self.name, [('ACTION', text)])
            return succeed(text)
        for line in text.split('\n'):
            self.account.client.say(self.name, line)
        return succeed(text)

    def leave(self):
        if False:
            while True:
                i = 10
        if self.account.client is None:
            raise locals.OfflineError
        self.account.client.leave(self.name)
        self.account.client.getGroupConversation(self.name, 1)

class IRCProto(basesupport.AbstractClientMixin, irc.IRCClient):

    def __init__(self, account, chatui, logonDeferred=None):
        if False:
            print('Hello World!')
        basesupport.AbstractClientMixin.__init__(self, account, chatui, logonDeferred)
        self._namreplies = {}
        self._ingroups = {}
        self._groups = {}
        self._topics = {}

    def getGroupConversation(self, name, hide=0):
        if False:
            while True:
                i = 10
        name = name.lower()
        return self.chat.getGroupConversation(self.chat.getGroup(name, self), stayHidden=hide)

    def getPerson(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.chat.getPerson(name, self)

    def connectionMade(self):
        if False:
            while True:
                i = 10
        try:
            self.performLogin = True
            self.nickname = self.account.username
            self.password = self.account.password
            self.realname = 'Twisted-IM user'
            irc.IRCClient.connectionMade(self)
            for channel in self.account.channels:
                self.joinGroup(channel)
            self.account._isOnline = 1
            if self._logonDeferred is not None:
                self._logonDeferred.callback(self)
            self.chat.getContactsList()
        except BaseException:
            import traceback
            traceback.print_exc()

    def setNick(self, nick):
        if False:
            print('Hello World!')
        self.name = nick
        self.accountName = '%s (IRC)' % nick
        irc.IRCClient.setNick(self, nick)

    def kickedFrom(self, channel, kicker, message):
        if False:
            print('Hello World!')
        '\n        Called when I am kicked from a channel.\n        '
        return self.chat.getGroupConversation(self.chat.getGroup(channel[1:], self), 1)

    def userKicked(self, kickee, channel, kicker, message):
        if False:
            print('Hello World!')
        pass

    def noticed(self, username, channel, message):
        if False:
            while True:
                i = 10
        self.privmsg(username, channel, message, {'dontAutoRespond': 1})

    def privmsg(self, username, channel, message, metadata=None):
        if False:
            i = 10
            return i + 15
        if metadata is None:
            metadata = {}
        username = username.split('!', 1)[0]
        if username == self.name:
            return
        if channel[0] == '#':
            group = channel[1:]
            self.getGroupConversation(group).showGroupMessage(username, message, metadata)
            return
        self.chat.getConversation(self.getPerson(username)).showMessage(message, metadata)

    def action(self, username, channel, emote):
        if False:
            while True:
                i = 10
        username = username.split('!', 1)[0]
        if username == self.name:
            return
        meta = {'style': 'emote'}
        if channel[0] == '#':
            group = channel[1:]
            self.getGroupConversation(group).showGroupMessage(username, emote, meta)
            return
        self.chat.getConversation(self.getPerson(username)).showMessage(emote, meta)

    def irc_RPL_NAMREPLY(self, prefix, params):
        if False:
            i = 10
            return i + 15
        '\n        RPL_NAMREPLY\n        >> NAMES #bnl\n        << :Arlington.VA.US.Undernet.Org 353 z3p = #bnl :pSwede Dan-- SkOyg AG\n        '
        group = params[2][1:].lower()
        users = params[3].split()
        for ui in range(len(users)):
            while users[ui][0] in ['@', '+']:
                users[ui] = users[ui][1:]
        if group not in self._namreplies:
            self._namreplies[group] = []
        self._namreplies[group].extend(users)
        for nickname in users:
            try:
                self._ingroups[nickname].append(group)
            except BaseException:
                self._ingroups[nickname] = [group]

    def irc_RPL_ENDOFNAMES(self, prefix, params):
        if False:
            print('Hello World!')
        group = params[1][1:]
        self.getGroupConversation(group).setGroupMembers(self._namreplies[group.lower()])
        del self._namreplies[group.lower()]

    def irc_RPL_TOPIC(self, prefix, params):
        if False:
            for i in range(10):
                print('nop')
        self._topics[params[1][1:]] = params[2]

    def irc_333(self, prefix, params):
        if False:
            i = 10
            return i + 15
        group = params[1][1:]
        self.getGroupConversation(group).setTopic(self._topics[group], params[2])
        del self._topics[group]

    def irc_TOPIC(self, prefix, params):
        if False:
            print('Hello World!')
        nickname = prefix.split('!')[0]
        group = params[0][1:]
        topic = params[1]
        self.getGroupConversation(group).setTopic(topic, nickname)

    def irc_JOIN(self, prefix, params):
        if False:
            while True:
                i = 10
        nickname = prefix.split('!')[0]
        group = params[0][1:].lower()
        if nickname != self.nickname:
            try:
                self._ingroups[nickname].append(group)
            except BaseException:
                self._ingroups[nickname] = [group]
            self.getGroupConversation(group).memberJoined(nickname)

    def irc_PART(self, prefix, params):
        if False:
            i = 10
            return i + 15
        nickname = prefix.split('!')[0]
        group = params[0][1:].lower()
        if nickname != self.nickname:
            if group in self._ingroups[nickname]:
                self._ingroups[nickname].remove(group)
                self.getGroupConversation(group).memberLeft(nickname)

    def irc_QUIT(self, prefix, params):
        if False:
            print('Hello World!')
        nickname = prefix.split('!')[0]
        if nickname in self._ingroups:
            for group in self._ingroups[nickname]:
                self.getGroupConversation(group).memberLeft(nickname)
            self._ingroups[nickname] = []

    def irc_NICK(self, prefix, params):
        if False:
            while True:
                i = 10
        fromNick = prefix.split('!')[0]
        toNick = params[0]
        if fromNick not in self._ingroups:
            return
        for group in self._ingroups[fromNick]:
            self.getGroupConversation(group).memberChangedNick(fromNick, toNick)
        self._ingroups[toNick] = self._ingroups[fromNick]
        del self._ingroups[fromNick]

    def irc_unknown(self, prefix, command, params):
        if False:
            for i in range(10):
                print('nop')
        pass

    def joinGroup(self, name):
        if False:
            return 10
        self.join(name)
        self.getGroupConversation(name)

@implementer(interfaces.IAccount)
class IRCAccount(basesupport.AbstractAccount):
    gatewayType = 'IRC'
    _groupFactory = IRCGroup
    _personFactory = IRCPerson

    def __init__(self, accountName, autoLogin, username, password, host, port, channels=''):
        if False:
            i = 10
            return i + 15
        basesupport.AbstractAccount.__init__(self, accountName, autoLogin, username, password, host, port)
        self.channels = [channel.strip() for channel in channels.split(',')]
        if self.channels == ['']:
            self.channels = []

    def _startLogOn(self, chatui):
        if False:
            print('Hello World!')
        logonDeferred = defer.Deferred()
        cc = protocol.ClientCreator(reactor, IRCProto, self, chatui, logonDeferred)
        d = cc.connectTCP(self.host, self.port)
        d.addErrback(logonDeferred.errback)
        return logonDeferred

    def logOff(self):
        if False:
            while True:
                i = 10
        pass