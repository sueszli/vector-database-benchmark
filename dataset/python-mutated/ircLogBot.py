"""
An example IRC log bot - logs a channel's events to a file.

If someone says the bot's name in the channel followed by a ':',
e.g.

    <foo> logbot: hello!

the bot will reply:

    <logbot> foo: I am a log bot

Run this script with two arguments, the channel name the bot should
connect to, and file to log to, e.g.:

    $ python ircLogBot.py test test.log

will log channel #test to the file 'test.log'.

To run the script:

    $ python ircLogBot.py <channel> <file>
"""
import sys
import time
from twisted.internet import protocol, reactor
from twisted.python import log
from twisted.words.protocols import irc

class MessageLogger:
    """
    An independent logger class (because separation of application
    and protocol logic is a good thing).
    """

    def __init__(self, file):
        if False:
            print('Hello World!')
        self.file = file

    def log(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Write a message to the file.'
        timestamp = time.strftime('[%H:%M:%S]', time.localtime(time.time()))
        self.file.write(f'{timestamp} {message}\n')
        self.file.flush()

    def close(self):
        if False:
            while True:
                i = 10
        self.file.close()

class LogBot(irc.IRCClient):
    """A logging IRC bot."""
    nickname = 'twistedbot'

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        irc.IRCClient.connectionMade(self)
        self.logger = MessageLogger(open(self.factory.filename, 'a'))
        self.logger.log('[connected at %s]' % time.asctime(time.localtime(time.time())))

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        irc.IRCClient.connectionLost(self, reason)
        self.logger.log('[disconnected at %s]' % time.asctime(time.localtime(time.time())))
        self.logger.close()

    def signedOn(self):
        if False:
            i = 10
            return i + 15
        'Called when bot has successfully signed on to server.'
        self.join(self.factory.channel)

    def joined(self, channel):
        if False:
            for i in range(10):
                print('nop')
        'This will get called when the bot joins the channel.'
        self.logger.log('[I have joined %s]' % channel)

    def privmsg(self, user, channel, msg):
        if False:
            for i in range(10):
                print('nop')
        'This will get called when the bot receives a message.'
        user = user.split('!', 1)[0]
        self.logger.log(f'<{user}> {msg}')
        if channel == self.nickname:
            msg = "It isn't nice to whisper!  Play nice with the group."
            self.msg(user, msg)
            return
        if msg.startswith(self.nickname + ':'):
            msg = '%s: I am a log bot' % user
            self.msg(channel, msg)
            self.logger.log(f'<{self.nickname}> {msg}')

    def action(self, user, channel, msg):
        if False:
            return 10
        'This will get called when the bot sees someone do an action.'
        user = user.split('!', 1)[0]
        self.logger.log(f'* {user} {msg}')

    def irc_NICK(self, prefix, params):
        if False:
            print('Hello World!')
        'Called when an IRC user changes their nickname.'
        old_nick = prefix.split('!')[0]
        new_nick = params[0]
        self.logger.log(f'{old_nick} is now known as {new_nick}')

    def alterCollidedNick(self, nickname):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate an altered version of a nickname that caused a collision in an\n        effort to create an unused related name for subsequent registration.\n        '
        return nickname + '^'

class LogBotFactory(protocol.ClientFactory):
    """A factory for LogBots.

    A new protocol instance will be created each time we connect to the server.
    """

    def __init__(self, channel, filename):
        if False:
            i = 10
            return i + 15
        self.channel = channel
        self.filename = filename

    def buildProtocol(self, addr):
        if False:
            while True:
                i = 10
        p = LogBot()
        p.factory = self
        return p

    def clientConnectionLost(self, connector, reason):
        if False:
            return 10
        'If we get disconnected, reconnect to server.'
        connector.connect()

    def clientConnectionFailed(self, connector, reason):
        if False:
            i = 10
            return i + 15
        print('connection failed:', reason)
        reactor.stop()
if __name__ == '__main__':
    log.startLogging(sys.stdout)
    f = LogBotFactory(sys.argv[1], sys.argv[2])
    reactor.connectTCP('irc.freenode.net', 6667, f)
    reactor.run()