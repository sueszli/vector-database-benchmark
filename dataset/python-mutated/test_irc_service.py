"""
Tests for IRC portions of L{twisted.words.service}.
"""
from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase

class IRCUserTests(IRCTestCase):
    """
    Isolated tests for L{IRCUser}
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets up a Realm, Portal, Factory, IRCUser, Transport, and Connection\n        for our tests.\n        '
        self.realm = InMemoryWordsRealm('example.com')
        self.checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.portal = portal.Portal(self.realm, [self.checker])
        self.checker.addUser('john', 'pass')
        self.factory = IRCFactory(self.realm, self.portal)
        self.ircUser = self.factory.buildProtocol(None)
        self.stringTransport = proto_helpers.StringTransport()
        self.ircUser.makeConnection(self.stringTransport)

    def test_sendMessage(self):
        if False:
            print('Hello World!')
        '\n        Sending a message to a user after they have sent NICK, but before they\n        have authenticated, results in a message from "example.com".\n        '
        self.ircUser.irc_NICK('', ['mynick'])
        self.stringTransport.clear()
        self.ircUser.sendMessage('foo')
        self.assertEqualBufferValue(self.stringTransport.value(), ':example.com foo mynick\r\n')

    def test_utf8Messages(self):
        if False:
            print('Hello World!')
        '\n        When a UTF8 message is sent with sendMessage and the current IRCUser\n        has a UTF8 nick and is set to UTF8 encoding, the message will be\n        written to the transport.\n        '
        expectedResult = ':example.com тест ник\r\n'.encode()
        self.ircUser.irc_NICK('', ['ник'.encode()])
        self.stringTransport.clear()
        self.ircUser.sendMessage('тест'.encode())
        self.assertEqualBufferValue(self.stringTransport.value(), expectedResult)

    def test_invalidEncodingNick(self):
        if False:
            i = 10
            return i + 15
        "\n        A NICK command sent with a nickname that cannot be decoded with the\n        current IRCUser's encoding results in a PRIVMSG from NickServ\n        indicating that the nickname could not be decoded.\n        "
        self.ircUser.irc_NICK('', [b'\xd4\xc5\xd3\xd4'])
        self.assertRaises(UnicodeError)

    def response(self):
        if False:
            i = 10
            return i + 15
        '\n        Grabs our responses and then clears the transport\n        '
        response = self.ircUser.transport.value()
        self.ircUser.transport.clear()
        if bytes != str and isinstance(response, bytes):
            response = response.decode('utf-8')
        response = response.splitlines()
        return [irc.parsemsg(r) for r in response]

    def scanResponse(self, response, messageType):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets messages out of a response\n\n        @param response: The parsed IRC messages of the response, as returned\n        by L{IRCUserTests.response}\n\n        @param messageType: The string type of the desired messages.\n\n        @return: An iterator which yields 2-tuples of C{(index, ircMessage)}\n        '
        for (n, message) in enumerate(response):
            if message[1] == messageType:
                yield (n, message)

    def test_sendNickSendsGreeting(self):
        if False:
            i = 10
            return i + 15
        '\n        Receiving NICK without authenticating sends the MOTD Start and MOTD End\n        messages, which is required by certain popular IRC clients (such as\n        Pidgin) before a connection is considered to be fully established.\n        '
        self.ircUser.irc_NICK('', ['mynick'])
        response = self.response()
        start = list(self.scanResponse(response, irc.RPL_MOTDSTART))
        end = list(self.scanResponse(response, irc.RPL_ENDOFMOTD))
        self.assertEqual(start, [(0, ('example.com', '375', ['mynick', '- example.com Message of the Day - ']))])
        self.assertEqual(end, [(1, ('example.com', '376', ['mynick', 'End of /MOTD command.']))])

    def test_fullLogin(self):
        if False:
            print('Hello World!')
        '\n        Receiving USER, PASS, NICK will log in the user, and transmit the\n        appropriate response messages.\n        '
        self.ircUser.irc_USER('', ['john doe'])
        self.ircUser.irc_PASS('', ['pass'])
        self.ircUser.irc_NICK('', ['john'])
        version = 'Your host is example.com, running version {}'.format(self.factory._serverInfo['serviceVersion'])
        creation = 'This server was created on {}'.format(self.factory._serverInfo['creationDate'])
        self.assertEqual(self.response(), [('example.com', '375', ['john', '- example.com Message of the Day - ']), ('example.com', '376', ['john', 'End of /MOTD command.']), ('example.com', '001', ['john', 'connected to Twisted IRC']), ('example.com', '002', ['john', version]), ('example.com', '003', ['john', creation]), ('example.com', '004', ['john', 'example.com', self.factory._serverInfo['serviceVersion'], 'w', 'n'])])

    def test_PART(self):
        if False:
            while True:
                i = 10
        '\n        irc_PART\n        '
        self.ircUser.irc_NICK('testuser', ['mynick'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.RPL_MOTDSTART)
        self.ircUser.irc_JOIN('testuser', ['somechannel'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.ERR_NOSUCHCHANNEL)
        self.ircUser.irc_PART('testuser', [b'somechannel', b'booga'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.ERR_NOTONCHANNEL)
        self.ircUser.irc_PART('testuser', ['somechannel', 'booga'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.ERR_NOTONCHANNEL)

    def test_NAMES(self):
        if False:
            return 10
        '\n        irc_NAMES\n        '
        self.ircUser.irc_NICK('', ['testuser'])
        self.ircUser.irc_JOIN('', ['somechannel'])
        self.ircUser.transport.clear()
        self.ircUser.irc_NAMES('', ['somechannel'])
        response = self.response()
        self.assertEqual(response[0][1], irc.RPL_ENDOFNAMES)

class MocksyIRCUser(IRCUser):

    def __init__(self):
        if False:
            print('Hello World!')
        self.realm = InMemoryWordsRealm('example.com')
        self.mockedCodes = []

    def sendMessage(self, code, *_, **__):
        if False:
            while True:
                i = 10
        self.mockedCodes.append(code)
BADTEXT = b'\xff'

class IRCUserBadEncodingTests(IRCTestCase):
    """
    Verifies that L{IRCUser} sends the correct error messages back to clients
    when given indecipherable bytes
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.ircUser = MocksyIRCUser()

    def assertChokesOnBadBytes(self, irc_x, error):
        if False:
            while True:
                i = 10
        "\n        Asserts that IRCUser sends the relevant error code when a given irc_x\n        dispatch method is given undecodable bytes.\n\n        @param irc_x: the name of the irc_FOO method to test.\n        For example, irc_x = 'PRIVMSG' will check irc_PRIVMSG\n\n        @param error: the error code irc_x should send. For example,\n        irc.ERR_NOTONCHANNEL\n        "
        getattr(self.ircUser, 'irc_%s' % irc_x)(None, [BADTEXT])
        self.assertEqual(self.ircUser.mockedCodes, [error])

    def test_JOIN(self):
        if False:
            while True:
                i = 10
        "\n        Tests that irc_JOIN sends ERR_NOSUCHCHANNEL if the channel name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('JOIN', irc.ERR_NOSUCHCHANNEL)

    def test_NAMES(self):
        if False:
            return 10
        "\n        Tests that irc_NAMES sends ERR_NOSUCHCHANNEL if the channel name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('NAMES', irc.ERR_NOSUCHCHANNEL)

    def test_TOPIC(self):
        if False:
            return 10
        "\n        Tests that irc_TOPIC sends ERR_NOSUCHCHANNEL if the channel name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('TOPIC', irc.ERR_NOSUCHCHANNEL)

    def test_LIST(self):
        if False:
            return 10
        "\n        Tests that irc_LIST sends ERR_NOSUCHCHANNEL if the channel name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('LIST', irc.ERR_NOSUCHCHANNEL)

    def test_MODE(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that irc_MODE sends ERR_NOSUCHNICK if the target name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('MODE', irc.ERR_NOSUCHNICK)

    def test_PRIVMSG(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that irc_PRIVMSG sends ERR_NOSUCHNICK if the target name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('PRIVMSG', irc.ERR_NOSUCHNICK)

    def test_WHOIS(self):
        if False:
            print('Hello World!')
        "\n        Tests that irc_WHOIS sends ERR_NOSUCHNICK if the target name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('WHOIS', irc.ERR_NOSUCHNICK)

    def test_PART(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that irc_PART sends ERR_NOTONCHANNEL if the target name can't\n        be decoded.\n        "
        self.assertChokesOnBadBytes('PART', irc.ERR_NOTONCHANNEL)

    def test_WHO(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that irc_WHO immediately ends the WHO list if the target name\n        can't be decoded.\n        "
        self.assertChokesOnBadBytes('WHO', irc.RPL_ENDOFWHO)