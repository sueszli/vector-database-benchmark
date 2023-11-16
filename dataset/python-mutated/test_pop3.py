"""
Test cases for L{twisted.mail.pop3} module.
"""
import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util

class UtilityTests(unittest.SynchronousTestCase):
    """
    Test the various helper functions and classes used by the POP3 server
    protocol implementation.
    """

    def test_LineBuffering(self):
        if False:
            return 10
        '\n        Test creating a LineBuffer and feeding it some lines.  The lines should\n        build up in its internal buffer for a while and then get spat out to\n        the writer.\n        '
        output = []
        input = iter(itertools.cycle(['012', '345', '6', '7', '8', '9']))
        c = pop3._IteratorBuffer(output.extend, input, 6)
        i = iter(c)
        self.assertEqual(output, [])
        next(i)
        self.assertEqual(output, [])
        next(i)
        self.assertEqual(output, [])
        next(i)
        self.assertEqual(output, ['012', '345', '6'])
        for n in range(5):
            next(i)
        self.assertEqual(output, ['012', '345', '6', '7', '8', '9', '012', '345'])

    def test_FinishLineBuffering(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that a LineBuffer flushes everything when its iterator is\n        exhausted, and itself raises StopIteration.\n        '
        output = []
        input = iter(['a', 'b', 'c'])
        c = pop3._IteratorBuffer(output.extend, input, 5)
        for i in c:
            pass
        self.assertEqual(output, ['a', 'b', 'c'])

    def test_SuccessResponseFormatter(self):
        if False:
            i = 10
            return i + 15
        "\n        Test that the thing that spits out POP3 'success responses' works\n        right.\n        "
        self.assertEqual(pop3.successResponse(b'Great.'), b'+OK Great.\r\n')

    def test_StatLineFormatter(self):
        if False:
            print('Hello World!')
        '\n        Test that the function which formats stat lines does so appropriately.\n        '
        statLine = list(pop3.formatStatResponse([]))[-1]
        self.assertEqual(statLine, b'+OK 0 0\r\n')
        statLine = list(pop3.formatStatResponse([10, 31, 0, 10101]))[-1]
        self.assertEqual(statLine, b'+OK 4 10142\r\n')

    def test_ListLineFormatter(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the function which formats the lines in response to a LIST\n        command does so appropriately.\n        '
        listLines = list(pop3.formatListResponse([]))
        self.assertEqual(listLines, [b'+OK 0\r\n', b'.\r\n'])
        listLines = list(pop3.formatListResponse([1, 2, 3, 100]))
        self.assertEqual(listLines, [b'+OK 4\r\n', b'1 1\r\n', b'2 2\r\n', b'3 3\r\n', b'4 100\r\n', b'.\r\n'])

    def test_UIDListLineFormatter(self):
        if False:
            while True:
                i = 10
        '\n        Test that the function which formats lines in response to a UIDL\n        command does so appropriately.\n        '
        uids = ['abc', 'def', 'ghi']
        listLines = list(pop3.formatUIDListResponse([], uids.__getitem__))
        self.assertEqual(listLines, [b'+OK \r\n', b'.\r\n'])
        listLines = list(pop3.formatUIDListResponse([123, 431, 591], uids.__getitem__))
        self.assertEqual(listLines, [b'+OK \r\n', b'1 abc\r\n', b'2 def\r\n', b'3 ghi\r\n', b'.\r\n'])
        listLines = list(pop3.formatUIDListResponse([0, None, 591], uids.__getitem__))
        self.assertEqual(listLines, [b'+OK \r\n', b'1 abc\r\n', b'3 ghi\r\n', b'.\r\n'])

class MyVirtualPOP3(mail.protocols.VirtualPOP3):
    """
    A virtual-domain-supporting POP3 server.
    """
    magic = b'<moshez>'

    def authenticateUserAPOP(self, user, digest):
        if False:
            while True:
                i = 10
        '\n        Authenticate against a user against a virtual domain.\n\n        @param user: The username.\n        @param digest: The digested password.\n\n        @return: A three-tuple like the one returned by\n            L{IRealm.requestAvatar}.  The mailbox will be for the user given\n            by C{user}.\n        '
        (user, domain) = self.lookupDomain(user)
        return self.service.domains[b'baz.com'].authenticateUserAPOP(user, digest, self.magic, domain)

class DummyDomain:
    """
    A virtual domain for a POP3 server.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.users = {}

    def addUser(self, name):
        if False:
            while True:
                i = 10
        '\n        Create a mailbox for a new user.\n\n        @param name: The username.\n        '
        self.users[name] = []

    def addMessage(self, name, message):
        if False:
            while True:
                i = 10
        '\n        Add a message to the mailbox of the named user.\n\n        @param name: The username.\n        @param message: The contents of the message.\n        '
        self.users[name].append(message)

    def authenticateUserAPOP(self, name, digest, magic, domain):
        if False:
            while True:
                i = 10
        '\n        Succeed with a L{ListMailbox}.\n\n        @param name: The name of the user authenticating.\n        @param digest: ignored\n        @param magic: ignored\n        @param domain: ignored\n\n        @return: A three-tuple like the one returned by\n            L{IRealm.requestAvatar}.  The mailbox will be for the user given\n            by C{name}.\n        '
        return (pop3.IMailbox, ListMailbox(self.users[name]), lambda : None)

class ListMailbox:
    """
    A simple in-memory list implementation of L{IMailbox}.
    """

    def __init__(self, list):
        if False:
            i = 10
            return i + 15
        '\n        @param list: The messages.\n        '
        self.list = list

    def listMessages(self, i=None):
        if False:
            while True:
                i = 10
        '\n        Get some message information.\n\n        @param i: See L{pop3.IMailbox.listMessages}.\n        @return: See L{pop3.IMailbox.listMessages}.\n        '
        if i is None:
            return [len(l) for l in self.list]
        return len(self.list[i])

    def getMessage(self, i):
        if False:
            i = 10
            return i + 15
        '\n        Get the message content.\n\n        @param i: See L{pop3.IMailbox.getMessage}.\n        @return: See L{pop3.IMailbox.getMessage}.\n        '
        return BytesIO(self.list[i])

    def getUidl(self, i):
        if False:
            i = 10
            return i + 15
        '\n        Construct a UID by using the given index value.\n\n        @param i: See L{pop3.IMailbox.getUidl}.\n        @return: See L{pop3.IMailbox.getUidl}.\n        '
        return i

    def deleteMessage(self, i):
        if False:
            i = 10
            return i + 15
        '\n        Wipe the message at the given index.\n\n        @param i: See L{pop3.IMailbox.deleteMessage}.\n        '
        self.list[i] = b''

    def sync(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        No-op.\n\n        @see: L{pop3.IMailbox.sync}\n        '

class MyPOP3Downloader(pop3.POP3Client):
    """
    A POP3 client which downloads all messages from the server.
    """

    def handle_WELCOME(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Authenticate.\n\n        @param line: The welcome response.\n        '
        pop3.POP3Client.handle_WELCOME(self, line)
        self.apop(b'hello@baz.com', b'world')

    def handle_APOP(self, line):
        if False:
            print('Hello World!')
        '\n        Require an I{OK} response to I{APOP}.\n\n        @param line: The I{APOP} response.\n        '
        parts = line.split()
        code = parts[0]
        if code != b'+OK':
            raise AssertionError(f'code is: {code} , parts is: {parts} ')
        self.lines = []
        self.retr(1)

    def handle_RETR_continue(self, line):
        if False:
            print('Hello World!')
        '\n        Record one line of message information.\n\n        @param line: A I{RETR} response line.\n        '
        self.lines.append(line)

    def handle_RETR_end(self):
        if False:
            while True:
                i = 10
        '\n        Record the received message information.\n        '
        self.message = b'\n'.join(self.lines) + b'\n'
        self.quit()

    def handle_QUIT(self, line):
        if False:
            while True:
                i = 10
        '\n        Require an I{OK} response to I{QUIT}.\n\n        @param line: The I{QUIT} response.\n        '
        if line[:3] != b'+OK':
            raise AssertionError(b'code is ' + line)

class POP3Tests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}.
    """
    message = b'Subject: urgent\n\nSomeone set up us the bomb!\n'
    expectedOutput = b'+OK <moshez>\r\n+OK Authentication succeeded\r\n+OK \r\n1 0\r\n.\r\n+OK %d\r\nSubject: urgent\r\n\r\nSomeone set up us the bomb!\r\n.\r\n+OK \r\n' % (len(message),)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up a POP3 server with virtual domain support.\n        '
        self.factory = internet.protocol.Factory()
        self.factory.domains = {}
        self.factory.domains[b'baz.com'] = DummyDomain()
        self.factory.domains[b'baz.com'].addUser(b'hello')
        self.factory.domains[b'baz.com'].addMessage(b'hello', self.message)

    def test_messages(self):
        if False:
            print('Hello World!')
        '\n        Messages can be downloaded over a loopback TCP connection.\n        '
        client = LineSendingProtocol([b'APOP hello@baz.com world', b'UIDL', b'RETR 1', b'QUIT'])
        server = MyVirtualPOP3()
        server.service = self.factory

        def check(ignored):
            if False:
                print('Hello World!')
            output = b'\r\n'.join(client.response) + b'\r\n'
            self.assertEqual(output, self.expectedOutput)
        return loopback.loopbackTCP(server, client).addCallback(check)

    def test_loopback(self):
        if False:
            while True:
                i = 10
        '\n        Messages can be downloaded over a loopback connection.\n        '
        protocol = MyVirtualPOP3()
        protocol.service = self.factory
        clientProtocol = MyPOP3Downloader()

        def check(ignored):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(clientProtocol.message, self.message)
            protocol.connectionLost(failure.Failure(Exception('Test harness disconnect')))
        d = loopback.loopbackAsync(protocol, clientProtocol)
        return d.addCallback(check)
    test_loopback.suppress = [util.suppress(message='twisted.mail.pop3.POP3Client is deprecated')]

    def test_incorrectDomain(self):
        if False:
            i = 10
            return i + 15
        '\n        Look up a user in a domain which this server does not support.\n        '
        factory = internet.protocol.Factory()
        factory.domains = {}
        factory.domains[b'twistedmatrix.com'] = DummyDomain()
        server = MyVirtualPOP3()
        server.service = factory
        exc = self.assertRaises(pop3.POP3Error, server.authenticateUserAPOP, b'nobody@baz.com', b'password')
        self.assertEqual(exc.args[0], 'no such domain baz.com')

class DummyPOP3(pop3.POP3):
    """
    A simple POP3 server with a hard-coded mailbox for any user.
    """
    magic = b'<moshez>'

    def authenticateUserAPOP(self, user, password):
        if False:
            print('Hello World!')
        '\n        Succeed with a L{DummyMailbox}.\n\n        @param user: ignored\n        @param password: ignored\n\n        @return: A three-tuple like the one returned by\n            L{IRealm.requestAvatar}.\n        '
        return (pop3.IMailbox, DummyMailbox(ValueError), lambda : None)

class DummyPOP3Auth(DummyPOP3):
    """
    Class to test successful authentication in twisted.mail.pop3.POP3.
    """

    def __init__(self, user, password):
        if False:
            while True:
                i = 10
        self.portal = cred.portal.Portal(TestRealm())
        ch = cred.checkers.InMemoryUsernamePasswordDatabaseDontUse()
        ch.addUser(user, password)
        self.portal.registerChecker(ch)

class DummyMailbox(pop3.Mailbox):
    """
    An in-memory L{pop3.IMailbox} implementation.

    @ivar messages: A sequence of L{bytes} defining the messages in this
        mailbox.

    @ivar exceptionType: The type of exception to raise when an out-of-bounds
        index is addressed.
    """
    messages = [b'From: moshe\nTo: moshe\n\nHow are you, friend?\n']

    def __init__(self, exceptionType):
        if False:
            for i in range(10):
                print('nop')
        self.messages = DummyMailbox.messages[:]
        self.exceptionType = exceptionType

    def listMessages(self, i=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get some message information.\n\n        @param i: See L{pop3.IMailbox.listMessages}.\n        @return: See L{pop3.IMailbox.listMessages}.\n        '
        if i is None:
            return [len(m) for m in self.messages]
        if i >= len(self.messages):
            raise self.exceptionType()
        return len(self.messages[i])

    def getMessage(self, i):
        if False:
            i = 10
            return i + 15
        '\n        Get the message content.\n\n        @param i: See L{pop3.IMailbox.getMessage}.\n        @return: See L{pop3.IMailbox.getMessage}.\n        '
        return BytesIO(self.messages[i])

    def getUidl(self, i):
        if False:
            print('Hello World!')
        '\n        Construct a UID which is simply the string representation of the given\n        index.\n\n        @param i: See L{pop3.IMailbox.getUidl}.\n        @return: See L{pop3.IMailbox.getUidl}.\n        '
        if i >= len(self.messages):
            raise self.exceptionType()
        return b'%d' % (i,)

    def deleteMessage(self, i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wipe the message at the given index.\n\n        @param i: See L{pop3.IMailbox.deleteMessage}.\n        '
        self.messages[i] = b''

class AnotherPOP3Tests(unittest.TestCase):
    """
    Additional L{pop3.POP3} tests.
    """

    def runTest(self, lines, expectedOutput, protocolInstance=None):
        if False:
            print('Hello World!')
        '\n        Assert that when C{lines} are delivered to L{pop3.POP3} it responds\n        with C{expectedOutput}.\n\n        @param lines: A sequence of L{bytes} representing lines to deliver to\n            the server.\n\n        @param expectedOutput: A sequence of L{bytes} representing the\n            expected response from the server.\n\n        @param protocolInstance: Instance of L{twisted.mail.pop3.POP3} or\n            L{None}. If L{None}, a new DummyPOP3 will be used.\n\n        @return: A L{Deferred} that fires when the lines have been delivered\n            and the output checked.\n        '
        dummy = protocolInstance if protocolInstance else DummyPOP3()
        client = LineSendingProtocol(lines)
        d = loopback.loopbackAsync(dummy, client)
        return d.addCallback(self._cbRunTest, client, dummy, expectedOutput)

    def _cbRunTest(self, ignored, client, dummy, expectedOutput):
        if False:
            print('Hello World!')
        self.assertEqual(b'\r\n'.join(expectedOutput), b'\r\n'.join(client.response))
        dummy.connectionLost(failure.Failure(Exception('Test harness disconnect')))
        return ignored

    def test_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a lot of different POP3 commands in an extremely pipelined\n        scenario.\n\n        This test may cover legitimate behavior, but the intent and\n        granularity are not very good.  It would likely be an improvement to\n        split it into a number of smaller, more focused tests.\n        '
        return self.runTest([b'APOP moshez dummy', b'LIST', b'UIDL', b'RETR 1', b'RETR 2', b'DELE 1', b'RETR 1', b'QUIT'], [b'+OK <moshez>', b'+OK Authentication succeeded', b'+OK 1', b'1 44', b'.', b'+OK ', b'1 0', b'.', b'+OK 44', b'From: moshe', b'To: moshe', b'', b'How are you, friend?', b'.', b'-ERR Bad message number argument', b'+OK ', b'-ERR message deleted', b'+OK '])

    def test_noop(self):
        if False:
            return 10
        '\n        Test the no-op command.\n        '
        return self.runTest([b'APOP spiv dummy', b'NOOP', b'QUIT'], [b'+OK <moshez>', b'+OK Authentication succeeded', b'+OK ', b'+OK '])

    def test_badUTF8CharactersInCommand(self):
        if False:
            print('Hello World!')
        '\n        Sending a command with invalid UTF-8 characters\n        will raise a L{pop3.POP3Error}.\n        '
        error = b'not authenticated yet: cannot do \x81PASS'
        d = self.runTest([b'\x81PASS', b'QUIT'], [b'+OK <moshez>', b'-ERR bad protocol or server: POP3Error: ' + error, b'+OK '])
        errors = self.flushLoggedErrors(pop3.POP3Error)
        self.assertEqual(len(errors), 1)
        return d

    def test_authListing(self):
        if False:
            while True:
                i = 10
        "\n        L{pop3.POP3} responds to an I{AUTH} command with a list of supported\n        authentication types based on its factory's C{challengers}.\n        "
        p = DummyPOP3()
        p.factory = internet.protocol.Factory()
        p.factory.challengers = {b'Auth1': None, b'secondAuth': None, b'authLast': None}
        client = LineSendingProtocol([b'AUTH', b'QUIT'])
        d = loopback.loopbackAsync(p, client)
        return d.addCallback(self._cbTestAuthListing, client)

    def _cbTestAuthListing(self, ignored, client):
        if False:
            print('Hello World!')
        self.assertTrue(client.response[1].startswith(b'+OK'))
        self.assertEqual(sorted(client.response[2:5]), [b'AUTH1', b'AUTHLAST', b'SECONDAUTH'])
        self.assertEqual(client.response[5], b'.')

    def run_PASS(self, real_user, real_password, tried_user=None, tried_password=None, after_auth_input=[], after_auth_output=[]):
        if False:
            while True:
                i = 10
        '\n        Test a login with PASS.\n\n        If L{real_user} matches L{tried_user} and L{real_password} matches\n        L{tried_password}, a successful login will be expected.\n        Otherwise an unsuccessful login will be expected.\n\n        @type real_user: L{bytes}\n        @param real_user: The user to test.\n\n        @type real_password: L{bytes}\n        @param real_password: The password of the test user.\n\n        @type tried_user: L{bytes} or L{None}\n        @param tried_user: The user to call USER with.\n            If None, real_user will be used.\n\n        @type tried_password: L{bytes} or L{None}\n        @param tried_password: The password to call PASS with.\n            If None, real_password will be used.\n\n        @type after_auth_input: L{list} of l{bytes}\n        @param after_auth_input: Extra protocol input after authentication.\n\n        @type after_auth_output: L{list} of l{bytes}\n        @param after_auth_output: Extra protocol output after authentication.\n        '
        if not tried_user:
            tried_user = real_user
        if not tried_password:
            tried_password = real_password
        response = [b'+OK <moshez>', b'+OK USER accepted, send PASS', b'-ERR Authentication failed']
        if real_user == tried_user and real_password == tried_password:
            response = [b'+OK <moshez>', b'+OK USER accepted, send PASS', b'+OK Authentication succeeded']
        fullInput = [b' '.join([b'USER', tried_user]), b' '.join([b'PASS', tried_password])]
        fullInput += after_auth_input + [b'QUIT']
        response += after_auth_output + [b'+OK ']
        return self.runTest(fullInput, response, protocolInstance=DummyPOP3Auth(real_user, real_password))

    def run_PASS_before_USER(self, password):
        if False:
            while True:
                i = 10
        '\n        Test protocol violation produced by calling PASS before USER.\n        @type password: L{bytes}\n        @param password: A password to test.\n        '
        return self.runTest([b' '.join([b'PASS', password]), b'QUIT'], [b'+OK <moshez>', b'-ERR USER required before PASS', b'+OK '])

    def test_illegal_PASS_before_USER(self):
        if False:
            while True:
                i = 10
        '\n        Test PASS before USER with a wrong password.\n        '
        return self.run_PASS_before_USER(b'fooz')

    def test_empty_PASS_before_USER(self):
        if False:
            return 10
        '\n        Test PASS before USER with an empty password.\n        '
        return self.run_PASS_before_USER(b'')

    def test_one_space_PASS_before_USER(self):
        if False:
            while True:
                i = 10
        '\n        Test PASS before USER with an password that is a space.\n        '
        return self.run_PASS_before_USER(b' ')

    def test_space_PASS_before_USER(self):
        if False:
            return 10
        '\n        Test PASS before USER with a password containing a space.\n        '
        return self.run_PASS_before_USER(b'fooz barz')

    def test_multiple_spaces_PASS_before_USER(self):
        if False:
            print('Hello World!')
        '\n        Test PASS before USER with a password containing multiple spaces.\n        '
        return self.run_PASS_before_USER(b'fooz barz asdf')

    def test_other_whitespace_PASS_before_USER(self):
        if False:
            return 10
        '\n        Test PASS before USER with a password containing tabs and spaces.\n        '
        return self.run_PASS_before_USER(b'fooz barz\tcrazy@! \t ')

    def test_good_PASS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test PASS with a good password.\n        '
        return self.run_PASS(b'testuser', b'fooz')

    def test_space_PASS(self):
        if False:
            i = 10
            return i + 15
        '\n        Test PASS with a password containing a space.\n        '
        return self.run_PASS(b'testuser', b'fooz barz')

    def test_multiple_spaces_PASS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test PASS with a password containing a space.\n        '
        return self.run_PASS(b'testuser', b'fooz barz asdf')

    def test_other_whitespace_PASS(self):
        if False:
            return 10
        '\n        Test PASS with a password containing tabs and spaces.\n        '
        return self.run_PASS(b'testuser', b'fooz barz\tcrazy@! \t ')

    def test_pass_wrong_user(self):
        if False:
            while True:
                i = 10
        '\n        Test PASS with a wrong user.\n        '
        return self.run_PASS(b'testuser', b'fooz', tried_user=b'wronguser')

    def test_wrong_PASS(self):
        if False:
            return 10
        '\n        Test PASS with a wrong password.\n        '
        return self.run_PASS(b'testuser', b'fooz', tried_password=b'barz')

    def test_wrong_space_PASS(self):
        if False:
            i = 10
            return i + 15
        '\n        Test PASS with a password containing a space.\n        '
        return self.run_PASS(b'testuser', b'fooz barz', tried_password=b'foozbarz ')

    def test_wrong_multiple_spaces_PASS(self):
        if False:
            while True:
                i = 10
        '\n        Test PASS with a password containing a space.\n        '
        return self.run_PASS(b'testuser', b'fooz barz asdf', tried_password=b'foozbarz   ')

    def test_wrong_other_whitespace_PASS(self):
        if False:
            while True:
                i = 10
        '\n        Test PASS with a password containing tabs and spaces.\n        '
        return self.run_PASS(b'testuser', b'fooz barz\tcrazy@! \t ')

    def test_wrong_command(self):
        if False:
            print('Hello World!')
        '\n        After logging in, test a dummy command that is not defined.\n        '
        extra_input = [b'DUMMY COMMAND']
        extra_output = [b' '.join([b'-ERR bad protocol or server: POP3Error:', b'Unknown protocol command: DUMMY'])]
        return self.run_PASS(b'testuser', b'testpassword', after_auth_input=extra_input, after_auth_output=extra_output).addCallback(self.flushLoggedErrors, pop3.POP3Error)

@implementer(pop3.IServerFactory)
class TestServerFactory:
    """
    A L{pop3.IServerFactory} implementation, for use by the test suite, with
    some behavior controlled by the values of (settable) public attributes and
    other behavior based on values hard-coded both here and in some test
    methods.
    """

    def cap_IMPLEMENTATION(self):
        if False:
            print('Hello World!')
        '\n        Return the hard-coded value.\n\n        @return: L{pop3.IServerFactory}\n        '
        return 'Test Implementation String'

    def cap_EXPIRE(self):
        if False:
            print('Hello World!')
        '\n        Return the hard-coded value.\n\n        @return: L{pop3.IServerFactory}\n        '
        return 60
    challengers = OrderedDict([(b'SCHEME_1', None), (b'SCHEME_2', None)])

    def cap_LOGIN_DELAY(self):
        if False:
            while True:
                i = 10
        '\n        Return the hard-coded value.\n\n        @return: L{pop3.IServerFactory}\n        '
        return 120
    pue = True

    def perUserExpiration(self):
        if False:
            while True:
                i = 10
        '\n        Return the hard-coded value.\n\n        @return: L{pop3.IServerFactory}\n        '
        return self.pue
    puld = True

    def perUserLoginDelay(self):
        if False:
            print('Hello World!')
        '\n        Return the hard-coded value.\n\n        @return: L{pop3.IServerFactory}\n        '
        return self.puld

class TestMailbox:
    """
    An incomplete L{IMailbox} implementation with certain per-user values
    hard-coded and known by tests in this module.


    This is useful for testing the server's per-user capability
    implementation.
    """
    loginDelay = 100
    messageExpiration = 25

def contained(testcase, s, *caps):
    if False:
        return 10
    '\n    Assert that the given capability is included in all of the capability\n    sets.\n\n    @param testcase: A L{unittest.TestCase} to use to make assertions.\n\n    @param s: The capability for which to check.\n    @type s: L{bytes}\n\n    @param caps: The capability sets in which to check.\n    @type caps: L{tuple} of iterable\n    '
    for c in caps:
        testcase.assertIn(s, c)

class CapabilityTests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}'s per-user capability handling.
    """

    def setUp(self):
        if False:
            return 10
        '\n        Create a POP3 server with some capabilities.\n        '
        s = BytesIO()
        p = pop3.POP3()
        p.factory = TestServerFactory()
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        p.do_CAPA()
        self.caps = p.listCapabilities()
        self.pcaps = s.getvalue().splitlines()
        s = BytesIO()
        p.mbox = TestMailbox()
        p.transport = internet.protocol.FileWrapper(s)
        p.do_CAPA()
        self.lpcaps = s.getvalue().splitlines()
        p.connectionLost(failure.Failure(Exception('Test harness disconnect')))

    def test_UIDL(self):
        if False:
            i = 10
            return i + 15
        '\n        The server can advertise the I{UIDL} capability.\n        '
        contained(self, b'UIDL', self.caps, self.pcaps, self.lpcaps)

    def test_TOP(self):
        if False:
            while True:
                i = 10
        '\n        The server can advertise the I{TOP} capability.\n        '
        contained(self, b'TOP', self.caps, self.pcaps, self.lpcaps)

    def test_USER(self):
        if False:
            i = 10
            return i + 15
        '\n        The server can advertise the I{USER} capability.\n        '
        contained(self, b'USER', self.caps, self.pcaps, self.lpcaps)

    def test_EXPIRE(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The server can advertise its per-user expiration as well as a global\n        expiration.\n        '
        contained(self, b'EXPIRE 60 USER', self.caps, self.pcaps)
        contained(self, b'EXPIRE 25', self.lpcaps)

    def test_IMPLEMENTATION(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The server can advertise its implementation string.\n        '
        contained(self, b'IMPLEMENTATION Test Implementation String', self.caps, self.pcaps, self.lpcaps)

    def test_SASL(self):
        if False:
            print('Hello World!')
        '\n        The server can advertise the SASL schemes it supports.\n        '
        contained(self, b'SASL SCHEME_1 SCHEME_2', self.caps, self.pcaps, self.lpcaps)

    def test_LOGIN_DELAY(self):
        if False:
            while True:
                i = 10
        '\n        The can advertise a per-user login delay as well as a global login\n        delay.\n        '
        contained(self, b'LOGIN-DELAY 120 USER', self.caps, self.pcaps)
        self.assertIn(b'LOGIN-DELAY 100', self.lpcaps)

class GlobalCapabilitiesTests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}'s global capability handling.
    """

    def setUp(self):
        if False:
            return 10
        '\n        Create a POP3 server with some capabilities.\n        '
        s = BytesIO()
        p = pop3.POP3()
        p.factory = TestServerFactory()
        p.factory.pue = p.factory.puld = False
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        p.do_CAPA()
        self.caps = p.listCapabilities()
        self.pcaps = s.getvalue().splitlines()
        s = BytesIO()
        p.mbox = TestMailbox()
        p.transport = internet.protocol.FileWrapper(s)
        p.do_CAPA()
        self.lpcaps = s.getvalue().splitlines()
        p.connectionLost(failure.Failure(Exception('Test harness disconnect')))

    def test_EXPIRE(self):
        if False:
            i = 10
            return i + 15
        "\n        I{EXPIRE} is in the server's advertised capabilities.\n        "
        contained(self, b'EXPIRE 60', self.caps, self.pcaps, self.lpcaps)

    def test_LOGIN_DELAY(self):
        if False:
            print('Hello World!')
        "\n        I{LOGIN-DELAY} is in the server's advertised capabilities.\n        "
        contained(self, b'LOGIN-DELAY 120', self.caps, self.pcaps, self.lpcaps)

class TestRealm:
    """
    An L{IRealm} which knows about a single test account's mailbox.
    """

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            return 10
        '\n        Retrieve a mailbox for I{testuser} or fail.\n\n        @param avatarId: See L{IRealm.requestAvatar}.\n        @param mind: See L{IRealm.requestAvatar}.\n        @param interfaces: See L{IRealm.requestAvatar}.\n\n        @raises: L{AssertionError} when requesting an C{avatarId} other than\n            I{testuser}.\n        '
        if avatarId == b'testuser':
            return (pop3.IMailbox, DummyMailbox(ValueError), lambda : None)
        assert False

class SASLTests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}'s SASL implementation.
    """

    def test_ValidLogin(self):
        if False:
            while True:
                i = 10
        "\n        A CRAM-MD5-based SASL login attempt succeeds if it uses a username and\n        a hashed password known to the server's credentials checker.\n        "
        p = pop3.POP3()
        p.factory = TestServerFactory()
        p.factory.challengers = {b'CRAM-MD5': cred.credentials.CramMD5Credentials}
        p.portal = cred.portal.Portal(TestRealm())
        ch = cred.checkers.InMemoryUsernamePasswordDatabaseDontUse()
        ch.addUser(b'testuser', b'testpassword')
        p.portal.registerChecker(ch)
        s = BytesIO()
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        p.lineReceived(b'CAPA')
        self.assertTrue(s.getvalue().find(b'SASL CRAM-MD5') >= 0)
        p.lineReceived(b'AUTH CRAM-MD5')
        chal = s.getvalue().splitlines()[-1][2:]
        chal = base64.b64decode(chal)
        response = hmac.HMAC(b'testpassword', chal, digestmod=md5).hexdigest().encode('ascii')
        p.lineReceived(base64.b64encode(b'testuser ' + response))
        self.assertTrue(p.mbox)
        self.assertTrue(s.getvalue().splitlines()[-1].find(b'+OK') >= 0)
        p.connectionLost(failure.Failure(Exception('Test harness disconnect')))

class CommandMixin:
    """
    Tests for all the commands a POP3 server is allowed to receive.
    """
    extraMessage = b'From: guy\nTo: fellow\n\nMore message text for you.\n'

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Make a POP3 server protocol instance hooked up to a simple mailbox and\n        a transport that buffers output to a BytesIO.\n        '
        p = pop3.POP3()
        p.mbox = self.mailboxType(self.exceptionType)
        p.schedule = list
        self.pop3Server = p
        s = BytesIO()
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        s.seek(0)
        s.truncate(0)
        self.pop3Transport = s

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        '\n        Disconnect the server protocol so it can clean up anything it might\n        need to clean up.\n        '
        self.pop3Server.connectionLost(failure.Failure(Exception('Test harness disconnect')))

    def _flush(self):
        if False:
            i = 10
            return i + 15
        '\n        Do some of the things that the reactor would take care of, if the\n        reactor were actually running.\n        '
        self.pop3Server.transport._checkProducer()

    def test_LIST(self):
        if False:
            while True:
                i = 10
        '\n        Test the two forms of list: with a message index number, which should\n        return a short-form response, and without a message index number, which\n        should return a long-form response, one line per message.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'LIST 1')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 1 44\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LIST')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 1\r\n1 44\r\n.\r\n')

    def test_LISTWithBadArgument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that non-integers and out-of-bound integers produce appropriate\n        error responses.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'LIST a')
        self.assertEqual(s.getvalue(), b'-ERR Invalid message-number: a\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LIST 0')
        self.assertEqual(s.getvalue(), b'-ERR Invalid message-number: 0\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LIST 2')
        self.assertEqual(s.getvalue(), b'-ERR Invalid message-number: 2\r\n')
        s.seek(0)
        s.truncate(0)

    def test_UIDL(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the two forms of the UIDL command.  These are just like the two\n        forms of the LIST command.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'UIDL 1')
        self.assertEqual(s.getvalue(), b'+OK 0\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'UIDL')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK \r\n1 0\r\n.\r\n')

    def test_UIDLWithBadArgument(self):
        if False:
            return 10
        '\n        Test that UIDL with a non-integer or an out-of-bounds integer produces\n        the appropriate error response.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'UIDL a')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'UIDL 0')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'UIDL 2')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)

    def test_STAT(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the single form of the STAT command, which returns a short-form\n        response of the number of messages in the mailbox and their total size.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'STAT')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 1 44\r\n')

    def test_RETR(self):
        if False:
            return 10
        '\n        Test downloading a message.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'RETR 1')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 44\r\nFrom: moshe\r\nTo: moshe\r\n\r\nHow are you, friend?\r\n.\r\n')
        s.seek(0)
        s.truncate(0)

    def test_RETRWithBadArgument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that trying to download a message with a bad argument, either not\n        an integer or an out-of-bounds integer, fails with the appropriate\n        error response.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'RETR a')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'RETR 0')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'RETR 2')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)

    def test_TOP(self):
        if False:
            i = 10
            return i + 15
        '\n        Test downloading the headers and part of the body of a message.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'TOP 1 0')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK Top of message follows\r\nFrom: moshe\r\nTo: moshe\r\n\r\n.\r\n')

    def test_TOPWithBadArgument(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that trying to download a message with a bad argument, either a\n        message number which isn't an integer or is an out-of-bounds integer or\n        a number of lines which isn't an integer or is a negative integer,\n        fails with the appropriate error response.\n        "
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'TOP 1 a')
        self.assertEqual(s.getvalue(), b'-ERR Bad line count argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP 1 -1')
        self.assertEqual(s.getvalue(), b'-ERR Bad line count argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP a 1')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP 0 1')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP 3 1')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)

    def test_LAST(self):
        if False:
            print('Hello World!')
        '\n        Test the exceedingly pointless LAST command, which tells you the\n        highest message index which you have already downloaded.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 0\r\n')
        s.seek(0)
        s.truncate(0)

    def test_RetrieveUpdatesHighest(self):
        if False:
            while True:
                i = 10
        '\n        Test that issuing a RETR command updates the LAST response.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'RETR 2')
        self._flush()
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 2\r\n')
        s.seek(0)
        s.truncate(0)

    def test_TopUpdatesHighest(self):
        if False:
            while True:
                i = 10
        '\n        Test that issuing a TOP command updates the LAST response.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'TOP 2 10')
        self._flush()
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 2\r\n')

    def test_HighestOnlyProgresses(self):
        if False:
            return 10
        "\n        Test that downloading a message with a smaller index than the current\n        LAST response doesn't change the LAST response.\n        "
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'RETR 2')
        self._flush()
        p.lineReceived(b'TOP 1 10')
        self._flush()
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 2\r\n')

    def test_ResetClearsHighest(self):
        if False:
            return 10
        '\n        Test that issuing RSET changes the LAST response to 0.\n        '
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'RETR 2')
        self._flush()
        p.lineReceived(b'RSET')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 0\r\n')
_listMessageDeprecation = 'twisted.mail.pop3.IMailbox.listMessages may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.'
_listMessageSuppression = util.suppress(message=_listMessageDeprecation, category=PendingDeprecationWarning)
_getUidlDeprecation = 'twisted.mail.pop3.IMailbox.getUidl may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.'
_getUidlSuppression = util.suppress(message=_getUidlDeprecation, category=PendingDeprecationWarning)

class IndexErrorCommandTests(CommandMixin, unittest.TestCase):
    """
    Run all of the command tests against a mailbox which raises IndexError
    when an out of bounds request is made.  This behavior will be deprecated
    shortly and then removed.
    """
    exceptionType = IndexError
    mailboxType = DummyMailbox

    def test_LISTWithBadArgument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An attempt to get metadata about a message with a bad argument fails\n        with an I{ERR} response even if the mailbox implementation raises\n        L{IndexError}.\n        '
        return CommandMixin.test_LISTWithBadArgument(self)
    test_LISTWithBadArgument.suppress = [_listMessageSuppression]

    def test_UIDLWithBadArgument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An attempt to look up the UID of a message with a bad argument fails\n        with an I{ERR} response even if the mailbox implementation raises\n        L{IndexError}.\n        '
        return CommandMixin.test_UIDLWithBadArgument(self)
    test_UIDLWithBadArgument.suppress = [_getUidlSuppression]

    def test_TOPWithBadArgument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An attempt to download some of a message with a bad argument fails with\n        an I{ERR} response even if the mailbox implementation raises\n        L{IndexError}.\n        '
        return CommandMixin.test_TOPWithBadArgument(self)
    test_TOPWithBadArgument.suppress = [_listMessageSuppression]

    def test_RETRWithBadArgument(self):
        if False:
            i = 10
            return i + 15
        '\n        An attempt to download a message with a bad argument fails with an\n        I{ERR} response even if the mailbox implementation raises\n        L{IndexError}.\n        '
        return CommandMixin.test_RETRWithBadArgument(self)
    test_RETRWithBadArgument.suppress = [_listMessageSuppression]

class ValueErrorCommandTests(CommandMixin, unittest.TestCase):
    """
    Run all of the command tests against a mailbox which raises ValueError
    when an out of bounds request is made.  This is the correct behavior and
    after support for mailboxes which raise IndexError is removed, this will
    become just C{CommandTestCase}.
    """
    exceptionType = ValueError
    mailboxType = DummyMailbox

class SyncDeferredMailbox(DummyMailbox):
    """
    Mailbox which has a listMessages implementation which returns a Deferred
    which has already fired.
    """

    def listMessages(self, n=None):
        if False:
            print('Hello World!')
        '\n        Synchronously list messages.\n\n        @type n: L{int} or L{None}\n        @param n: The 0-based index of the message.\n\n        @return: A L{Deferred} which already has a message list result.\n        '
        return defer.succeed(DummyMailbox.listMessages(self, n))

class IndexErrorSyncDeferredCommandTests(IndexErrorCommandTests):
    """
    Run all of the L{IndexErrorCommandTests} tests with a
    synchronous-Deferred returning IMailbox implementation.
    """
    mailboxType = SyncDeferredMailbox

class ValueErrorSyncDeferredCommandTests(ValueErrorCommandTests):
    """
    Run all of the L{ValueErrorCommandTests} tests with a
    synchronous-Deferred returning IMailbox implementation.
    """
    mailboxType = SyncDeferredMailbox

class AsyncDeferredMailbox(DummyMailbox):
    """
    Mailbox which has a listMessages implementation which returns a Deferred
    which has not yet fired.
    """

    def __init__(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        self.waiting = []
        DummyMailbox.__init__(self, *a, **kw)

    def listMessages(self, n=None):
        if False:
            return 10
        '\n        Record a new unfired L{Deferred} in C{self.waiting} and return it.\n\n        @type n: L{int} or L{None}\n        @param n: The 0-based index of the message.\n\n        @return: The L{Deferred}\n        '
        d = defer.Deferred()
        self.waiting.append((d, DummyMailbox.listMessages(self, n)))
        return d

class IndexErrorAsyncDeferredCommandTests(IndexErrorCommandTests):
    """
    Run all of the L{IndexErrorCommandTests} tests with an
    asynchronous-Deferred returning IMailbox implementation.
    """
    mailboxType = AsyncDeferredMailbox

    def _flush(self):
        if False:
            return 10
        "\n        Fire whatever Deferreds we've built up in our mailbox.\n        "
        while self.pop3Server.mbox.waiting:
            (d, a) = self.pop3Server.mbox.waiting.pop()
            d.callback(a)
        IndexErrorCommandTests._flush(self)

class ValueErrorAsyncDeferredCommandTests(ValueErrorCommandTests):
    """
    Run all of the L{IndexErrorCommandTests} tests with an
    asynchronous-Deferred returning IMailbox implementation.
    """
    mailboxType = AsyncDeferredMailbox

    def _flush(self):
        if False:
            print('Hello World!')
        "\n        Fire whatever Deferreds we've built up in our mailbox.\n        "
        while self.pop3Server.mbox.waiting:
            (d, a) = self.pop3Server.mbox.waiting.pop()
            d.callback(a)
        ValueErrorCommandTests._flush(self)

class POP3MiscTests(unittest.SynchronousTestCase):
    """
    Miscellaneous tests more to do with module/package structure than
    anything to do with the Post Office Protocol.
    """

    def test_all(self):
        if False:
            print('Hello World!')
        '\n        This test checks that all names listed in\n        twisted.mail.pop3.__all__ are actually present in the module.\n        '
        mod = twisted.mail.pop3
        for attr in mod.__all__:
            self.assertTrue(hasattr(mod, attr))

class POP3ClientDeprecationTests(unittest.SynchronousTestCase):
    """
    Tests for the now deprecated L{twisted.mail.pop3client} module.
    """

    def test_deprecation(self):
        if False:
            return 10
        '\n        A deprecation warning is emitted when directly importing the now\n        deprected pop3client module.\n\n        This test might fail is some other code has already imported it.\n        No code should use the deprected module.\n        '
        from twisted.mail import pop3client
        warningsShown = self.flushWarnings(offendingFunctions=[self.test_deprecation])
        self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], 'twisted.mail.pop3client was deprecated in Twisted 21.2.0. Use twisted.mail.pop3 instead.')
        self.assertEqual(len(warningsShown), 1)
        pop3client

class APOPCredentialsTests(unittest.SynchronousTestCase):

    def test_implementsIUsernamePassword(self):
        if False:
            return 10
        '\n        L{APOPCredentials} implements\n        L{twisted.cred.credentials.IUsernameHashedPassword}.\n        '
        self.assertTrue(verifyClass(IUsernameHashedPassword, pop3.APOPCredentials))