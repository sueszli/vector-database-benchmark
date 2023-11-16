"""
Post-office Protocol version 3.

@author: Glyph Lefkowitz
@author: Jp Calderone
"""
import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import IMailboxPOP3 as IMailbox, IServerFactoryPOP3 as IServerFactory
from twisted.protocols import basic, policies
from twisted.python import log

@implementer(cred.credentials.IUsernameHashedPassword)
class APOPCredentials:
    """
    Credentials for use in APOP authentication.

    @ivar magic: See L{__init__}
    @ivar username: See L{__init__}
    @ivar digest: See L{__init__}
    """

    def __init__(self, magic, username, digest):
        if False:
            while True:
                i = 10
        "\n        @type magic: L{bytes}\n        @param magic: The challenge string used to encrypt the password.\n\n        @type username: L{bytes}\n        @param username: The username associated with these credentials.\n\n        @type digest: L{bytes}\n        @param digest: An encrypted version of the user's password.  Should be\n            generated as an MD5 hash of the challenge string concatenated with\n            the plaintext password.\n        "
        self.magic = magic
        self.username = username
        self.digest = digest

    def checkPassword(self, password):
        if False:
            print('Hello World!')
        '\n        Validate a plaintext password against the credentials.\n\n        @type password: L{bytes}\n        @param password: A plaintext password.\n\n        @rtype: L{bool}\n        @return: C{True} if the credentials represented by this object match\n        the given password, C{False} if they do not.\n        '
        seed = self.magic + password
        myDigest = md5(seed).hexdigest()
        return myDigest == self.digest

class _HeadersPlusNLines:
    """
    A utility class to retrieve the header and some lines of the body of a mail
    message.

    @ivar _file: See L{__init__}
    @ivar _extraLines: See L{__init__}

    @type linecount: L{int}
    @ivar linecount: The number of full lines of the message body scanned.

    @type headers: L{bool}
    @ivar headers: An indication of which part of the message is being scanned.
        C{True} for the header and C{False} for the body.

    @type done: L{bool}
    @ivar done: A flag indicating when the desired part of the message has been
        scanned.

    @type buf: L{bytes}
    @ivar buf: The portion of the message body that has been scanned, up to
        C{n} lines.
    """

    def __init__(self, file, extraLines):
        if False:
            i = 10
            return i + 15
        '\n        @type file: file-like object\n        @param file: A file containing a mail message.\n\n        @type extraLines: L{int}\n        @param extraLines: The number of lines of the message body to retrieve.\n        '
        self._file = file
        self._extraLines = extraLines
        self.linecount = 0
        self.headers = 1
        self.done = 0
        self.buf = b''

    def read(self, bytes):
        if False:
            i = 10
            return i + 15
        '\n        Scan bytes from the file.\n\n        @type bytes: L{int}\n        @param bytes: The number of bytes to read from the file.\n\n        @rtype: L{bytes}\n        @return: Each portion of the header as it is scanned.  Then, full lines\n            of the message body as they are scanned.  When more than one line\n            of the header and/or body has been scanned, the result is the\n            concatenation of the lines.  When the scan results in no full\n            lines, the empty string is returned.\n        '
        if self.done:
            return b''
        data = self._file.read(bytes)
        if not data:
            return data
        if self.headers:
            (df, sz) = (data.find(b'\r\n\r\n'), 4)
            if df == -1:
                (df, sz) = (data.find(b'\n\n'), 2)
            if df != -1:
                df += sz
                val = data[:df]
                data = data[df:]
                self.linecount = 1
                self.headers = 0
        else:
            val = b''
        if self.linecount > 0:
            dsplit = (self.buf + data).split(b'\n')
            self.buf = dsplit[-1]
            for ln in dsplit[:-1]:
                if self.linecount > self._extraLines:
                    self.done = 1
                    return val
                val += ln + b'\n'
                self.linecount += 1
            return val
        else:
            return data

class _IteratorBuffer:
    """
    An iterator which buffers the elements of a container and periodically
    passes them as input to a writer.

    @ivar write: See L{__init__}.
    @ivar memoryBufferSize: See L{__init__}.

    @type bufSize: L{int}
    @ivar bufSize: The number of bytes currently in the buffer.

    @type lines: L{list} of L{bytes}
    @ivar lines: The buffer, which is a list of strings.

    @type iterator: iterator which yields L{bytes}
    @ivar iterator: An iterator over a container of strings.
    """
    bufSize = 0

    def __init__(self, write, iterable, memoryBufferSize=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        @type write: callable that takes L{list} of L{bytes}\n        @param write: A writer which is a callable that takes a list of\n            strings.\n\n        @type iterable: iterable which yields L{bytes}\n        @param iterable: An iterable container of strings.\n\n        @type memoryBufferSize: L{int} or L{None}\n        @param memoryBufferSize: The number of bytes to buffer before flushing\n            the buffer to the writer.\n        '
        self.lines = []
        self.write = write
        self.iterator = iter(iterable)
        if memoryBufferSize is None:
            memoryBufferSize = 2 ** 16
        self.memoryBufferSize = memoryBufferSize

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an iterator.\n\n        @rtype: iterator which yields L{bytes}\n        @return: An iterator over strings.\n        '
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the next string from the container, buffer it, and possibly send\n        the buffer to the writer.\n\n        The contents of the buffer are written when it is full or when no\n        further values are available from the container.\n\n        @raise StopIteration: When no further values are available from the\n        container.\n        '
        try:
            v = next(self.iterator)
        except StopIteration:
            if self.lines:
                self.write(self.lines)
            del self.iterator, self.lines, self.write
            raise
        else:
            if v is not None:
                self.lines.append(v)
                self.bufSize += len(v)
                if self.bufSize > self.memoryBufferSize:
                    self.write(self.lines)
                    self.lines = []
                    self.bufSize = 0
    next = __next__

def iterateLineGenerator(proto, gen):
    if False:
        while True:
            i = 10
    '\n    Direct the output of an iterator to the transport of a protocol and arrange\n    for iteration to take place.\n\n    @type proto: L{POP3}\n    @param proto: A POP3 server protocol.\n\n    @type gen: iterator which yields L{bytes}\n    @param gen: An iterator over strings.\n\n    @rtype: L{Deferred <defer.Deferred>}\n    @return: A deferred which fires when the iterator finishes.\n    '
    coll = _IteratorBuffer(proto.transport.writeSequence, gen)
    return proto.schedule(coll)

def successResponse(response):
    if False:
        i = 10
        return i + 15
    '\n    Format an object as a positive response.\n\n    @type response: stringifyable L{object}\n    @param response: An object with a string representation.\n\n    @rtype: L{bytes}\n    @return: A positive POP3 response string.\n    '
    if not isinstance(response, bytes):
        response = str(response).encode('utf-8')
    return b'+OK ' + response + b'\r\n'

def formatStatResponse(msgs):
    if False:
        print('Hello World!')
    '\n    Format a list of message sizes into a STAT response.\n\n    This generator function is intended to be used with\n    L{Cooperator <twisted.internet.task.Cooperator>}.\n\n    @type msgs: L{list} of L{int}\n    @param msgs: A list of message sizes.\n\n    @rtype: L{None} or L{bytes}\n    @return: Yields none until a result is available, then a string that is\n        suitable for use in a STAT response. The string consists of the number\n        of messages and the total size of the messages in octets.\n    '
    i = 0
    bytes = 0
    for size in msgs:
        i += 1
        bytes += size
        yield None
    yield successResponse(b'%d %d' % (i, bytes))

def formatListLines(msgs):
    if False:
        while True:
            i = 10
    '\n    Format a list of message sizes for use in a LIST response.\n\n    @type msgs: L{list} of L{int}\n    @param msgs: A list of message sizes.\n\n    @rtype: L{bytes}\n    @return: Yields a series of strings that are suitable for use as scan\n        listings in a LIST response. Each string consists of a message number\n        and its size in octets.\n    '
    i = 0
    for size in msgs:
        i += 1
        yield (b'%d %d\r\n' % (i, size))

def formatListResponse(msgs):
    if False:
        while True:
            i = 10
    '\n    Format a list of message sizes into a complete LIST response.\n\n    This generator function is intended to be used with\n    L{Cooperator <twisted.internet.task.Cooperator>}.\n\n    @type msgs: L{list} of L{int}\n    @param msgs: A list of message sizes.\n\n    @rtype: L{bytes}\n    @return: Yields a series of strings which make up a complete LIST response.\n    '
    yield successResponse(b'%d' % (len(msgs),))
    yield from formatListLines(msgs)
    yield b'.\r\n'

def formatUIDListLines(msgs, getUidl):
    if False:
        while True:
            i = 10
    '\n    Format a list of message sizes for use in a UIDL response.\n\n    @param msgs: See L{formatUIDListResponse}\n    @param getUidl: See L{formatUIDListResponse}\n\n    @rtype: L{bytes}\n    @return: Yields a series of strings that are suitable for use as unique-id\n        listings in a UIDL response. Each string consists of a message number\n        and its unique id.\n    '
    for (i, m) in enumerate(msgs):
        if m is not None:
            uid = getUidl(i)
            if not isinstance(uid, bytes):
                uid = str(uid).encode('utf-8')
            yield (b'%d %b\r\n' % (i + 1, uid))

def formatUIDListResponse(msgs, getUidl):
    if False:
        for i in range(10):
            print('nop')
    '\n    Format a list of message sizes into a complete UIDL response.\n\n    This generator function is intended to be used with\n    L{Cooperator <twisted.internet.task.Cooperator>}.\n\n    @type msgs: L{list} of L{int}\n    @param msgs: A list of message sizes.\n\n    @type getUidl: one-argument callable returning bytes\n    @param getUidl: A callable which takes a message index number and returns\n        the UID of the corresponding message in the mailbox.\n\n    @rtype: L{bytes}\n    @return: Yields a series of strings which make up a complete UIDL response.\n    '
    yield successResponse('')
    yield from formatUIDListLines(msgs, getUidl)
    yield b'.\r\n'

@implementer(interfaces.IProducer)
class POP3(basic.LineOnlyReceiver, policies.TimeoutMixin):
    """
    A POP3 server protocol.

    @type portal: L{Portal}
    @ivar portal: A portal for authentication.

    @type factory: L{IServerFactory} provider
    @ivar factory: A server factory which provides an interface for querying
        capabilities of the server.

    @type timeOut: L{int}
    @ivar timeOut: The number of seconds to wait for a command from the client
        before disconnecting.

    @type schedule: callable that takes interator and returns
        L{Deferred <defer.Deferred>}
    @ivar schedule: A callable that arranges for an iterator to be
        cooperatively iterated over along with all other iterators which have
        been passed to it such that runtime is divided between all of them.  It
        returns a deferred which fires when the iterator finishes.

    @type magic: L{bytes} or L{None}
    @ivar magic: An APOP challenge.  If not set, an APOP challenge string
        will be generated when a connection is made.

    @type _userIs: L{bytes} or L{None}
    @ivar _userIs: The username sent with the USER command.

    @type _onLogout: no-argument callable or L{None}
    @ivar _onLogout: The function to be executed when the connection is
        lost.

    @type mbox: L{IMailbox} provider
    @ivar mbox: The mailbox for the authenticated user.

    @type state: L{bytes}
    @ivar state: The state which indicates what type of messages are expected
        from the client.  Valid states are 'COMMAND' and 'AUTH'

    @type blocked: L{None} or L{list} of 2-L{tuple} of
        (E{1}) L{bytes} (E{2}) L{tuple} of L{bytes}
    @ivar blocked: A list of blocked commands.  While a response to a command
        is being generated by the server, other commands are blocked.  When
        no command is outstanding, C{blocked} is set to none.  Otherwise, it
        contains a list of information about blocked commands.  Each list
        entry consists of the command and the arguments to the command.

    @type _highest: L{int}
    @ivar _highest: The 1-based index of the highest message retrieved.

    @type _auth: L{IUsernameHashedPassword
        <cred.credentials.IUsernameHashedPassword>} provider
    @ivar _auth: Authorization credentials.
    """
    magic: Optional[bytes] = None
    _userIs = None
    _onLogout = None
    AUTH_CMDS = [b'CAPA', b'USER', b'PASS', b'APOP', b'AUTH', b'RPOP', b'QUIT']
    portal = None
    factory = None
    mbox = None
    timeOut = 300
    state = 'COMMAND'
    blocked = None
    schedule = staticmethod(task.coiterate)
    _highest = 0

    def connectionMade(self):
        if False:
            return 10
        '\n        Send a greeting to the client after the connection has been made.\n        '
        if self.magic is None:
            self.magic = self.generateMagic()
        self.successResponse(self.magic)
        self.setTimeout(self.timeOut)
        if getattr(self.factory, 'noisy', True):
            log.msg('New connection from ' + str(self.transport.getPeer()))

    def connectionLost(self, reason):
        if False:
            return 10
        '\n        Clean up when the connection has been lost.\n\n        @type reason: L{Failure}\n        @param reason: The reason the connection was terminated.\n        '
        if self._onLogout is not None:
            self._onLogout()
            self._onLogout = None
        self.setTimeout(None)

    def generateMagic(self):
        if False:
            while True:
                i = 10
        '\n        Generate an APOP challenge.\n\n        @rtype: L{bytes}\n        @return: An RFC 822 message id format string.\n        '
        return smtp.messageid()

    def successResponse(self, message=''):
        if False:
            i = 10
            return i + 15
        '\n        Send a response indicating success.\n\n        @type message: stringifyable L{object}\n        @param message: An object whose string representation should be\n            included in the response.\n        '
        self.transport.write(successResponse(message))

    def failResponse(self, message=b''):
        if False:
            i = 10
            return i + 15
        '\n        Send a response indicating failure.\n\n        @type message: stringifyable L{object}\n        @param message: An object whose string representation should be\n            included in the response.\n        '
        if not isinstance(message, bytes):
            message = str(message).encode('utf-8')
        self.sendLine(b'-ERR ' + message)

    def lineReceived(self, line):
        if False:
            print('Hello World!')
        '\n        Pass a received line to a state machine function.\n\n        @type line: L{bytes}\n        @param line: A received line.\n        '
        self.resetTimeout()
        getattr(self, 'state_' + self.state)(line)

    def _unblock(self, _):
        if False:
            return 10
        '\n        Process as many blocked commands as possible.\n\n        If there are no more blocked commands, set up for the next command to\n        be sent immediately.\n\n        @type _: L{object}\n        @param _: Ignored.\n        '
        commands = self.blocked
        self.blocked = None
        while commands and self.blocked is None:
            (cmd, args) = commands.pop(0)
            self.processCommand(cmd, *args)
        if self.blocked is not None:
            self.blocked.extend(commands)

    def state_COMMAND(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle received lines for the COMMAND state in which commands from the\n        client are expected.\n\n        @type line: L{bytes}\n        @param line: A received command.\n        '
        try:
            return self.processCommand(*line.split(b' '))
        except (ValueError, AttributeError, POP3Error, TypeError) as e:
            log.err()
            self.failResponse(b': '.join([b'bad protocol or server', e.__class__.__name__.encode('utf-8'), b''.join(e.args)]))

    def processCommand(self, command, *args):
        if False:
            for i in range(10):
                print('nop')
        "\n        Dispatch a command from the client for handling.\n\n        @type command: L{bytes}\n        @param command: A POP3 command.\n\n        @type args: L{tuple} of L{bytes}\n        @param args: Arguments to the command.\n\n        @raise POP3Error: When the command is invalid or the command requires\n            prior authentication which hasn't been performed.\n        "
        if self.blocked is not None:
            self.blocked.append((command, args))
            return
        command = command.upper()
        authCmd = command in self.AUTH_CMDS
        if not self.mbox and (not authCmd):
            raise POP3Error(b'not authenticated yet: cannot do ' + command)
        f = getattr(self, 'do_' + command.decode('utf-8'), None)
        if f:
            return f(*args)
        raise POP3Error(b'Unknown protocol command: ' + command)

    def listCapabilities(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a list of server capabilities suitable for use in a CAPA\n        response.\n\n        @rtype: L{list} of L{bytes}\n        @return: A list of server capabilities.\n        '
        baseCaps = [b'TOP', b'USER', b'UIDL', b'PIPELINE', b'CELERITY', b'AUSPEX', b'POTENCE']
        if IServerFactory.providedBy(self.factory):
            try:
                v = self.factory.cap_IMPLEMENTATION()
                if v and (not isinstance(v, bytes)):
                    v = str(v).encode('utf-8')
            except NotImplementedError:
                pass
            except BaseException:
                log.err()
            else:
                baseCaps.append(b'IMPLEMENTATION ' + v)
            try:
                v = self.factory.cap_EXPIRE()
                if v and (not isinstance(v, bytes)):
                    v = str(v).encode('utf-8')
            except NotImplementedError:
                pass
            except BaseException:
                log.err()
            else:
                if v is None:
                    v = b'NEVER'
                if self.factory.perUserExpiration():
                    if self.mbox:
                        v = str(self.mbox.messageExpiration).encode('utf-8')
                    else:
                        v = v + b' USER'
                baseCaps.append(b'EXPIRE ' + v)
            try:
                v = self.factory.cap_LOGIN_DELAY()
                if v and (not isinstance(v, bytes)):
                    v = str(v).encode('utf-8')
            except NotImplementedError:
                pass
            except BaseException:
                log.err()
            else:
                if self.factory.perUserLoginDelay():
                    if self.mbox:
                        v = str(self.mbox.loginDelay).encode('utf-8')
                    else:
                        v = v + b' USER'
                baseCaps.append(b'LOGIN-DELAY ' + v)
            try:
                v = self.factory.challengers
            except AttributeError:
                pass
            except BaseException:
                log.err()
            else:
                baseCaps.append(b'SASL ' + b' '.join(v.keys()))
        return baseCaps

    def do_CAPA(self):
        if False:
            while True:
                i = 10
        '\n        Handle a CAPA command.\n\n        Respond with the server capabilities.\n        '
        self.successResponse(b'I can do the following:')
        for cap in self.listCapabilities():
            self.sendLine(cap)
        self.sendLine(b'.')

    def do_AUTH(self, args=None):
        if False:
            print('Hello World!')
        '\n        Handle an AUTH command.\n\n        If the AUTH extension is not supported, send an error response.  If an\n        authentication mechanism was not specified in the command, send a list\n        of all supported authentication methods.  Otherwise, send an\n        authentication challenge to the client and transition to the\n        AUTH state.\n\n        @type args: L{bytes} or L{None}\n        @param args: The name of an authentication mechanism.\n        '
        if not getattr(self.factory, 'challengers', None):
            self.failResponse(b'AUTH extension unsupported')
            return
        if args is None:
            self.successResponse('Supported authentication methods:')
            for a in self.factory.challengers:
                self.sendLine(a.upper())
            self.sendLine(b'.')
            return
        auth = self.factory.challengers.get(args.strip().upper())
        if not self.portal or not auth:
            self.failResponse(b'Unsupported SASL selected')
            return
        self._auth = auth()
        chal = self._auth.getChallenge()
        self.sendLine(b'+ ' + base64.b64encode(chal))
        self.state = 'AUTH'

    def state_AUTH(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle received lines for the AUTH state in which an authentication\n        challenge response from the client is expected.\n\n        Transition back to the COMMAND state.  Check the credentials and\n        complete the authorization process with the L{_cbMailbox}\n        callback function on success or the L{_ebMailbox} and L{_ebUnexpected}\n        errback functions on failure.\n\n        @type line: L{bytes}\n        @param line: The challenge response.\n        '
        self.state = 'COMMAND'
        try:
            parts = base64.b64decode(line).split(None, 1)
        except binascii.Error:
            self.failResponse(b'Invalid BASE64 encoding')
        else:
            if len(parts) != 2:
                self.failResponse(b'Invalid AUTH response')
                return
            self._auth.username = parts[0]
            self._auth.response = parts[1]
            d = self.portal.login(self._auth, None, IMailbox)
            d.addCallback(self._cbMailbox, parts[0])
            d.addErrback(self._ebMailbox)
            d.addErrback(self._ebUnexpected)

    def do_APOP(self, user, digest):
        if False:
            i = 10
            return i + 15
        '\n        Handle an APOP command.\n\n        Perform APOP authentication and complete the authorization process with\n        the L{_cbMailbox} callback function on success or the L{_ebMailbox}\n        and L{_ebUnexpected} errback functions on failure.\n\n        @type user: L{bytes}\n        @param user: A username.\n\n        @type digest: L{bytes}\n        @param digest: An MD5 digest string.\n        '
        d = defer.maybeDeferred(self.authenticateUserAPOP, user, digest)
        d.addCallbacks(self._cbMailbox, self._ebMailbox, callbackArgs=(user,)).addErrback(self._ebUnexpected)

    def _cbMailbox(self, result, user):
        if False:
            while True:
                i = 10
        '\n        Complete successful authentication.\n\n        Save the mailbox and logout function for the authenticated user and\n        send a successful response to the client.\n\n        @type result: C{tuple}\n        @param result: The first item of the tuple is a\n            C{zope.interface.Interface} which is the interface\n            supported by the avatar.  The second item of the tuple is a\n            L{IMailbox} provider which is the mailbox for the\n            authenticated user.  The third item of the tuple is a no-argument\n            callable which is a function to be invoked when the session is\n            terminated.\n\n        @type user: L{bytes}\n        @param user: The user being authenticated.\n        '
        (interface, avatar, logout) = result
        if interface is not IMailbox:
            self.failResponse(b'Authentication failed')
            log.err('_cbMailbox() called with an interface other than IMailbox')
            return
        self.mbox = avatar
        self._onLogout = logout
        self.successResponse('Authentication succeeded')
        if getattr(self.factory, 'noisy', True):
            log.msg(b'Authenticated login for ' + user)

    def _ebMailbox(self, failure):
        if False:
            i = 10
            return i + 15
        '\n        Handle an expected authentication failure.\n\n        Send an appropriate error response for a L{LoginDenied} or\n        L{LoginFailed} authentication failure.\n\n        @type failure: L{Failure}\n        @param failure: The authentication error.\n        '
        failure = failure.trap(cred.error.LoginDenied, cred.error.LoginFailed)
        if issubclass(failure, cred.error.LoginDenied):
            self.failResponse('Access denied: ' + str(failure))
        elif issubclass(failure, cred.error.LoginFailed):
            self.failResponse(b'Authentication failed')
        if getattr(self.factory, 'noisy', True):
            log.msg('Denied login attempt from ' + str(self.transport.getPeer()))

    def _ebUnexpected(self, failure):
        if False:
            i = 10
            return i + 15
        '\n        Handle an unexpected authentication failure.\n\n        Send an error response for an unexpected authentication failure.\n\n        @type failure: L{Failure}\n        @param failure: The authentication error.\n        '
        self.failResponse('Server error: ' + failure.getErrorMessage())
        log.err(failure)

    def do_USER(self, user):
        if False:
            while True:
                i = 10
        '\n        Handle a USER command.\n\n        Save the username and send a successful response prompting the client\n        for the password.\n\n        @type user: L{bytes}\n        @param user: A username.\n        '
        self._userIs = user
        self.successResponse(b'USER accepted, send PASS')

    def do_PASS(self, password, *words):
        if False:
            while True:
                i = 10
        '\n        Handle a PASS command.\n\n        If a USER command was previously received, authenticate the user and\n        complete the authorization process with the L{_cbMailbox} callback\n        function on success or the L{_ebMailbox} and L{_ebUnexpected} errback\n        functions on failure.  If a USER command was not previously received,\n        send an error response.\n\n        @type password: L{bytes}\n        @param password: A password.\n\n        @type words: L{tuple} of L{bytes}\n        @param words: Other parts of the password split by spaces.\n        '
        if self._userIs is None:
            self.failResponse(b'USER required before PASS')
            return
        user = self._userIs
        self._userIs = None
        password = b' '.join((password,) + words)
        d = defer.maybeDeferred(self.authenticateUserPASS, user, password)
        d.addCallbacks(self._cbMailbox, self._ebMailbox, callbackArgs=(user,)).addErrback(self._ebUnexpected)

    def _longOperation(self, d):
        if False:
            while True:
                i = 10
        '\n        Stop timeouts and block further command processing while a long\n        operation completes.\n\n        @type d: L{Deferred <defer.Deferred>}\n        @param d: A deferred which triggers at the completion of a long\n            operation.\n\n        @rtype: L{Deferred <defer.Deferred>}\n        @return: A deferred which triggers after command processing resumes and\n            timeouts restart after the completion of a long operation.\n        '
        timeOut = self.timeOut
        self.setTimeout(None)
        self.blocked = []
        d.addCallback(self._unblock)
        d.addCallback(lambda ign: self.setTimeout(timeOut))
        return d

    def _coiterate(self, gen):
        if False:
            for i in range(10):
                print('nop')
        '\n        Direct the output of an iterator to the transport and arrange for\n        iteration to take place.\n\n        @type gen: iterable which yields L{bytes}\n        @param gen: An iterator over strings.\n\n        @rtype: L{Deferred <defer.Deferred>}\n        @return: A deferred which fires when the iterator finishes.\n        '
        return self.schedule(_IteratorBuffer(self.transport.writeSequence, gen))

    def do_STAT(self):
        if False:
            return 10
        '\n        Handle a STAT command.\n\n        @rtype: L{Deferred <defer.Deferred>}\n        @return: A deferred which triggers after the response to the STAT\n            command has been issued.\n        '
        d = defer.maybeDeferred(self.mbox.listMessages)

        def cbMessages(msgs):
            if False:
                for i in range(10):
                    print('nop')
            return self._coiterate(formatStatResponse(msgs))

        def ebMessages(err):
            if False:
                i = 10
                return i + 15
            self.failResponse(err.getErrorMessage())
            log.msg('Unexpected do_STAT failure:')
            log.err(err)
        return self._longOperation(d.addCallbacks(cbMessages, ebMessages))

    def do_LIST(self, i=None):
        if False:
            return 10
        '\n        Handle a LIST command.\n\n        @type i: L{bytes} or L{None}\n        @param i: A 1-based message index.\n\n        @rtype: L{Deferred <defer.Deferred>}\n        @return: A deferred which triggers after the response to the LIST\n            command has been issued.\n        '
        if i is None:
            d = defer.maybeDeferred(self.mbox.listMessages)

            def cbMessages(msgs):
                if False:
                    print('Hello World!')
                return self._coiterate(formatListResponse(msgs))

            def ebMessages(err):
                if False:
                    i = 10
                    return i + 15
                self.failResponse(err.getErrorMessage())
                log.msg('Unexpected do_LIST failure:')
                log.err(err)
            return self._longOperation(d.addCallbacks(cbMessages, ebMessages))
        else:
            try:
                i = int(i)
                if i < 1:
                    raise ValueError()
            except ValueError:
                if not isinstance(i, bytes):
                    i = str(i).encode('utf-8')
                self.failResponse(b'Invalid message-number: ' + i)
            else:
                d = defer.maybeDeferred(self.mbox.listMessages, i - 1)

                def cbMessage(msg):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.successResponse(b'%d %d' % (i, msg))

                def ebMessage(err):
                    if False:
                        i = 10
                        return i + 15
                    errcls = err.check(ValueError, IndexError)
                    if errcls is not None:
                        if errcls is IndexError:
                            warnings.warn('twisted.mail.pop3.IMailbox.listMessages may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
                        invalidNum = i
                        if invalidNum and (not isinstance(invalidNum, bytes)):
                            invalidNum = str(invalidNum).encode('utf-8')
                        self.failResponse(b'Invalid message-number: ' + invalidNum)
                    else:
                        self.failResponse(err.getErrorMessage())
                        log.msg('Unexpected do_LIST failure:')
                        log.err(err)
                d.addCallbacks(cbMessage, ebMessage)
                return self._longOperation(d)

    def do_UIDL(self, i=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle a UIDL command.\n\n        @type i: L{bytes} or L{None}\n        @param i: A 1-based message index.\n\n        @rtype: L{Deferred <defer.Deferred>}\n        @return: A deferred which triggers after the response to the UIDL\n            command has been issued.\n        '
        if i is None:
            d = defer.maybeDeferred(self.mbox.listMessages)

            def cbMessages(msgs):
                if False:
                    for i in range(10):
                        print('nop')
                return self._coiterate(formatUIDListResponse(msgs, self.mbox.getUidl))

            def ebMessages(err):
                if False:
                    return 10
                self.failResponse(err.getErrorMessage())
                log.msg('Unexpected do_UIDL failure:')
                log.err(err)
            return self._longOperation(d.addCallbacks(cbMessages, ebMessages))
        else:
            try:
                i = int(i)
                if i < 1:
                    raise ValueError()
            except ValueError:
                self.failResponse('Bad message number argument')
            else:
                try:
                    msg = self.mbox.getUidl(i - 1)
                except IndexError:
                    warnings.warn('twisted.mail.pop3.IMailbox.getUidl may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
                    self.failResponse('Bad message number argument')
                except ValueError:
                    self.failResponse('Bad message number argument')
                else:
                    if not isinstance(msg, bytes):
                        msg = str(msg).encode('utf-8')
                    self.successResponse(msg)

    def _getMessageFile(self, i):
        if False:
            print('Hello World!')
        '\n        Retrieve the size and contents of a message.\n\n        @type i: L{bytes}\n        @param i: A 1-based message index.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            2-L{tuple} of (E{1}) L{int}, (E{2}) file-like object\n        @return: A deferred which successfully fires with the size of the\n            message and a file containing the contents of the message.\n        '
        try:
            msg = int(i) - 1
            if msg < 0:
                raise ValueError()
        except ValueError:
            self.failResponse('Bad message number argument')
            return defer.succeed(None)
        sizeDeferred = defer.maybeDeferred(self.mbox.listMessages, msg)

        def cbMessageSize(size):
            if False:
                i = 10
                return i + 15
            if not size:
                return defer.fail(_POP3MessageDeleted())
            fileDeferred = defer.maybeDeferred(self.mbox.getMessage, msg)
            fileDeferred.addCallback(lambda fObj: (size, fObj))
            return fileDeferred

        def ebMessageSomething(err):
            if False:
                while True:
                    i = 10
            errcls = err.check(_POP3MessageDeleted, ValueError, IndexError)
            if errcls is _POP3MessageDeleted:
                self.failResponse('message deleted')
            elif errcls in (ValueError, IndexError):
                if errcls is IndexError:
                    warnings.warn('twisted.mail.pop3.IMailbox.listMessages may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
                self.failResponse('Bad message number argument')
            else:
                log.msg('Unexpected _getMessageFile failure:')
                log.err(err)
            return None
        sizeDeferred.addCallback(cbMessageSize)
        sizeDeferred.addErrback(ebMessageSomething)
        return sizeDeferred

    def _sendMessageContent(self, i, fpWrapper, successResponse):
        if False:
            i = 10
            return i + 15
        '\n        Send the contents of a message.\n\n        @type i: L{bytes}\n        @param i: A 1-based message index.\n\n        @type fpWrapper: callable that takes a file-like object and returns\n            a file-like object\n        @param fpWrapper:\n\n        @type successResponse: callable that takes L{int} and returns\n            L{bytes}\n        @param successResponse:\n\n        @rtype: L{Deferred}\n        @return: A deferred which triggers after the message has been sent.\n        '
        d = self._getMessageFile(i)

        def cbMessageFile(info):
            if False:
                i = 10
                return i + 15
            if info is None:
                return
            self._highest = max(self._highest, int(i))
            (resp, fp) = info
            fp = fpWrapper(fp)
            self.successResponse(successResponse(resp))
            s = basic.FileSender()
            d = s.beginFileTransfer(fp, self.transport, self.transformChunk)

            def cbFileTransfer(lastsent):
                if False:
                    while True:
                        i = 10
                if lastsent != b'\n':
                    line = b'\r\n.'
                else:
                    line = b'.'
                self.sendLine(line)

            def ebFileTransfer(err):
                if False:
                    i = 10
                    return i + 15
                self.transport.loseConnection()
                log.msg('Unexpected error in _sendMessageContent:')
                log.err(err)
            d.addCallback(cbFileTransfer)
            d.addErrback(ebFileTransfer)
            return d
        return self._longOperation(d.addCallback(cbMessageFile))

    def do_TOP(self, i, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle a TOP command.\n\n        @type i: L{bytes}\n        @param i: A 1-based message index.\n\n        @type size: L{bytes}\n        @param size: The number of lines of the message to retrieve.\n\n        @rtype: L{Deferred}\n        @return: A deferred which triggers after the response to the TOP\n            command has been issued.\n        '
        try:
            size = int(size)
            if size < 0:
                raise ValueError
        except ValueError:
            self.failResponse('Bad line count argument')
        else:
            return self._sendMessageContent(i, lambda fp: _HeadersPlusNLines(fp, size), lambda size: 'Top of message follows')

    def do_RETR(self, i):
        if False:
            print('Hello World!')
        '\n        Handle a RETR command.\n\n        @type i: L{bytes}\n        @param i: A 1-based message index.\n\n        @rtype: L{Deferred}\n        @return: A deferred which triggers after the response to the RETR\n            command has been issued.\n        '
        return self._sendMessageContent(i, lambda fp: fp, lambda size: '%d' % (size,))

    def transformChunk(self, chunk):
        if False:
            print('Hello World!')
        "\n        Transform a chunk of a message to POP3 message format.\n\n        Make sure each line ends with C{'\\r\\n'} and byte-stuff the\n        termination character (C{'.'}) by adding an extra one when one appears\n        at the beginning of a line.\n\n        @type chunk: L{bytes}\n        @param chunk: A string to transform.\n\n        @rtype: L{bytes}\n        @return: The transformed string.\n        "
        return chunk.replace(b'\n', b'\r\n').replace(b'\r\n.', b'\r\n..')

    def finishedFileTransfer(self, lastsent):
        if False:
            while True:
                i = 10
        '\n        Send the termination sequence.\n\n        @type lastsent: L{bytes}\n        @param lastsent: The last character of the file.\n        '
        if lastsent != b'\n':
            line = b'\r\n.'
        else:
            line = b'.'
        self.sendLine(line)

    def do_DELE(self, i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle a DELE command.\n\n        Mark a message for deletion and issue a successful response.\n\n        @type i: L{int}\n        @param i: A 1-based message index.\n        '
        i = int(i) - 1
        self.mbox.deleteMessage(i)
        self.successResponse()

    def do_NOOP(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle a NOOP command.\n\n        Do nothing but issue a successful response.\n        '
        self.successResponse()

    def do_RSET(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle a RSET command.\n\n        Unmark any messages that have been flagged for deletion.\n        '
        try:
            self.mbox.undeleteMessages()
        except BaseException:
            log.err()
            self.failResponse()
        else:
            self._highest = 0
            self.successResponse()

    def do_LAST(self):
        if False:
            i = 10
            return i + 15
        '\n        Handle a LAST command.\n\n        Respond with the 1-based index of the highest retrieved message.\n        '
        self.successResponse(self._highest)

    def do_RPOP(self, user):
        if False:
            return 10
        '\n        Handle an RPOP command.\n\n        RPOP is not supported.  Send an error response.\n\n        @type user: L{bytes}\n        @param user: A username.\n\n        '
        self.failResponse('permission denied, sucker')

    def do_QUIT(self):
        if False:
            while True:
                i = 10
        '\n        Handle a QUIT command.\n\n        Remove any messages marked for deletion, issue a successful response,\n        and drop the connection.\n        '
        if self.mbox:
            self.mbox.sync()
        self.successResponse()
        self.transport.loseConnection()

    def authenticateUserAPOP(self, user, digest):
        if False:
            while True:
                i = 10
        '\n        Perform APOP authentication.\n\n        @type user: L{bytes}\n        @param user: The name of the user attempting to log in.\n\n        @type digest: L{bytes}\n        @param digest: The challenge response.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully results in\n            3-L{tuple} of (E{1}) L{IMailbox <pop3.IMailbox>}, (E{2})\n            L{IMailbox <pop3.IMailbox>} provider, (E{3}) no-argument callable\n        @return: A deferred which fires when authentication is complete.  If\n            successful, it returns an L{IMailbox <pop3.IMailbox>} interface, a\n            mailbox, and a function to be invoked with the session is\n            terminated.  If authentication fails, the deferred fails with an\n            L{UnathorizedLogin <cred.error.UnauthorizedLogin>} error.\n\n        @raise cred.error.UnauthorizedLogin: When authentication fails.\n        '
        if self.portal is not None:
            return self.portal.login(APOPCredentials(self.magic, user, digest), None, IMailbox)
        raise cred.error.UnauthorizedLogin()

    def authenticateUserPASS(self, user, password):
        if False:
            i = 10
            return i + 15
        '\n        Perform authentication for a username/password login.\n\n        @type user: L{bytes}\n        @param user: The name of the user attempting to log in.\n\n        @type password: L{bytes}\n        @param password: The password to authenticate with.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully results in\n            3-L{tuple} of (E{1}) L{IMailbox <pop3.IMailbox>}, (E{2}) L{IMailbox\n            <pop3.IMailbox>} provider, (E{3}) no-argument callable\n        @return: A deferred which fires when authentication is complete.  If\n            successful, it returns a L{pop3.IMailbox} interface, a mailbox,\n            and a function to be invoked with the session is terminated.\n            If authentication fails, the deferred fails with an\n            L{UnathorizedLogin <cred.error.UnauthorizedLogin>} error.\n\n        @raise cred.error.UnauthorizedLogin: When authentication fails.\n        '
        if self.portal is not None:
            return self.portal.login(cred.credentials.UsernamePassword(user, password), None, IMailbox)
        raise cred.error.UnauthorizedLogin()

    def stopProducing(self):
        if False:
            return 10
        raise NotImplementedError()

@implementer(IMailbox)
class Mailbox:
    """
    A base class for mailboxes.
    """

    def listMessages(self, i=None):
        if False:
            while True:
                i = 10
        '\n        Retrieve the size of a message, or, if none is specified, the size of\n        each message in the mailbox.\n\n        @type i: L{int} or L{None}\n        @param i: The 0-based index of the message.\n\n        @rtype: L{int}, sequence of L{int}, or L{Deferred <defer.Deferred>}\n        @return: The number of octets in the specified message, or, if an\n            index is not specified, a sequence of the number of octets for\n            all messages in the mailbox or a deferred which fires with\n            one of those. Any value which corresponds to a deleted message\n            is set to 0.\n\n        @raise ValueError or IndexError: When the index does not correspond to\n            a message in the mailbox.  The use of ValueError is preferred.\n        '
        return []

    def getMessage(self, i):
        if False:
            while True:
                i = 10
        '\n        Retrieve a file containing the contents of a message.\n\n        @type i: L{int}\n        @param i: The 0-based index of a message.\n\n        @rtype: file-like object\n        @return: A file containing the message.\n\n        @raise ValueError or IndexError: When the index does not correspond to\n            a message in the mailbox.  The use of ValueError is preferred.\n        '
        raise ValueError

    def getUidl(self, i):
        if False:
            while True:
                i = 10
        '\n        Get a unique identifier for a message.\n\n        @type i: L{int}\n        @param i: The 0-based index of a message.\n\n        @rtype: L{bytes}\n        @return: A string of printable characters uniquely identifying the\n            message for all time.\n\n        @raise ValueError or IndexError: When the index does not correspond to\n            a message in the mailbox.  The use of ValueError is preferred.\n        '
        raise ValueError

    def deleteMessage(self, i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mark a message for deletion.\n\n        This must not change the number of messages in this mailbox.  Further\n        requests for the size of the deleted message should return 0.  Further\n        requests for the message itself may raise an exception.\n\n        @type i: L{int}\n        @param i: The 0-based index of a message.\n\n        @raise ValueError or IndexError: When the index does not correspond to\n            a message in the mailbox.  The use of ValueError is preferred.\n        '
        raise ValueError

    def undeleteMessages(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Undelete all messages marked for deletion.\n\n        Any message which can be undeleted should be returned to its original\n        position in the message sequence and retain its original UID.\n        '
        pass

    def sync(self):
        if False:
            i = 10
            return i + 15
        '\n        Discard the contents of any message marked for deletion.\n        '
        pass
(NONE, SHORT, FIRST_LONG, LONG) = range(4)
NEXT = {}
NEXT[NONE] = NONE
NEXT[SHORT] = NONE
NEXT[FIRST_LONG] = LONG
NEXT[LONG] = NONE

class POP3Client(basic.LineOnlyReceiver):
    """
    A POP3 client protocol.

    @type mode: L{int}
    @ivar mode: The type of response expected from the server.  Choices include
    none (0), a one line response (1), the first line of a multi-line
    response (2), and subsequent lines of a multi-line response (3).

    @type command: L{bytes}
    @ivar command: The command most recently sent to the server.

    @type welcomeRe: L{Pattern <re.Pattern.search>}
    @ivar welcomeRe: A regular expression which matches the APOP challenge in
        the server greeting.

    @type welcomeCode: L{bytes}
    @ivar welcomeCode: The APOP challenge passed in the server greeting.
    """
    mode = SHORT
    command = b'WELCOME'
    import re
    welcomeRe = re.compile(b'<(.*)>')

    def __init__(self):
        if False:
            return 10
        '\n        Issue deprecation warning.\n        '
        import warnings
        warnings.warn('twisted.mail.pop3.POP3Client is deprecated, please use twisted.mail.pop3.AdvancedPOP3Client instead.', DeprecationWarning, stacklevel=3)

    def sendShort(self, command, params=None):
        if False:
            return 10
        '\n        Send a POP3 command to which a short response is expected.\n\n        @type command: L{bytes}\n        @param command: A POP3 command.\n\n        @type params: stringifyable L{object} or L{None}\n        @param params: Command arguments.\n        '
        if params is not None:
            if not isinstance(params, bytes):
                params = str(params).encode('utf-8')
            self.sendLine(command + b' ' + params)
        else:
            self.sendLine(command)
        self.command = command
        self.mode = SHORT

    def sendLong(self, command, params):
        if False:
            i = 10
            return i + 15
        '\n        Send a POP3 command to which a long response is expected.\n\n        @type command: L{bytes}\n        @param command: A POP3 command.\n\n        @type params: stringifyable L{object}\n        @param params: Command arguments.\n        '
        if params:
            if not isinstance(params, bytes):
                params = str(params).encode('utf-8')
            self.sendLine(command + b' ' + params)
        else:
            self.sendLine(command)
        self.command = command
        self.mode = FIRST_LONG

    def handle_default(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle responses from the server for which no other handler exists.\n\n        @type line: L{bytes}\n        @param line: A received line.\n        '
        if line[:-4] == b'-ERR':
            self.mode = NONE

    def handle_WELCOME(self, line):
        if False:
            print('Hello World!')
        '\n        Handle a server response which is expected to be a server greeting.\n\n        @type line: L{bytes}\n        @param line: A received line.\n        '
        (code, data) = line.split(b' ', 1)
        if code != b'+OK':
            self.transport.loseConnection()
        else:
            m = self.welcomeRe.match(line)
            if m:
                self.welcomeCode = m.group(1)

    def _dispatch(self, command, default, *args):
        if False:
            while True:
                i = 10
        '\n        Dispatch a response from the server for handling.\n\n        Command X is dispatched to handle_X() if it exists.  If not, it is\n        dispatched to the default handler.\n\n        @type command: L{bytes}\n        @param command: The command.\n\n        @type default: callable that takes L{bytes} or\n            L{None}\n        @param default: The default handler.\n\n        @type args: L{tuple} or L{None}\n        @param args: Arguments to the handler function.\n        '
        try:
            method = getattr(self, 'handle_' + command.decode('utf-8'), default)
            if method is not None:
                method(*args)
        except BaseException:
            log.err()

    def lineReceived(self, line):
        if False:
            while True:
                i = 10
        '\n        Dispatch a received line for processing.\n\n        The choice of function to handle the received line is based on the\n        type of response expected to the command sent to the server and how\n        much of that response has been received.\n\n        An expected one line response to command X is handled by handle_X().\n        The first line of a multi-line response to command X is also handled by\n        handle_X().  Subsequent lines of the multi-line response are handled by\n        handle_X_continue() except for the last line which is handled by\n        handle_X_end().\n\n        @type line: L{bytes}\n        @param line: A received line.\n        '
        if self.mode == SHORT or self.mode == FIRST_LONG:
            self.mode = NEXT[self.mode]
            self._dispatch(self.command, self.handle_default, line)
        elif self.mode == LONG:
            if line == b'.':
                self.mode = NEXT[self.mode]
                self._dispatch(self.command + b'_end', None)
                return
            if line[:1] == b'.':
                line = line[1:]
            self._dispatch(self.command + b'_continue', None, line)

    def apopAuthenticate(self, user, password, magic):
        if False:
            i = 10
            return i + 15
        '\n        Perform an authenticated login.\n\n        @type user: L{bytes}\n        @param user: The username with which to log in.\n\n        @type password: L{bytes}\n        @param password: The password with which to log in.\n\n        @type magic: L{bytes}\n        @param magic: The challenge provided by the server.\n        '
        digest = md5(magic + password).hexdigest().encode('ascii')
        self.apop(user, digest)

    def apop(self, user, digest):
        if False:
            return 10
        '\n        Send an APOP command to perform authenticated login.\n\n        @type user: L{bytes}\n        @param user: The username with which to log in.\n\n        @type digest: L{bytes}\n        @param digest: The challenge response with which to authenticate.\n        '
        self.sendLong(b'APOP', b' '.join((user, digest)))

    def retr(self, i):
        if False:
            print('Hello World!')
        '\n        Send a RETR command to retrieve a message from the server.\n\n        @type i: L{int} or L{bytes}\n        @param i: A 0-based message index.\n        '
        self.sendLong(b'RETR', i)

    def dele(self, i):
        if False:
            i = 10
            return i + 15
        '\n        Send a DELE command to delete a message from the server.\n\n        @type i: L{int} or L{bytes}\n        @param i: A 0-based message index.\n        '
        self.sendShort(b'DELE', i)

    def list(self, i=''):
        if False:
            print('Hello World!')
        '\n        Send a LIST command to retrieve the size of a message or, if no message\n        is specified, the sizes of all messages.\n\n        @type i: L{int} or L{bytes}\n        @param i: A 0-based message index or the empty string to specify all\n            messages.\n        '
        self.sendLong(b'LIST', i)

    def uidl(self, i=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a UIDL command to retrieve the unique identifier of a message or,\n        if no message is specified, the unique identifiers of all messages.\n\n        @type i: L{int} or L{bytes}\n        @param i: A 0-based message index or the empty string to specify all\n            messages.\n        '
        self.sendLong(b'UIDL', i)

    def user(self, name):
        if False:
            print('Hello World!')
        '\n        Send a USER command to perform the first half of a plaintext login.\n\n        @type name: L{bytes}\n        @param name: The username with which to log in.\n        '
        self.sendShort(b'USER', name)

    def password(self, password):
        if False:
            while True:
                i = 10
        '\n        Perform the second half of a plaintext login.\n\n        @type password: L{bytes}\n        @param password: The plaintext password with which to authenticate.\n        '
        self.sendShort(b'PASS', password)
    pass_ = password

    def quit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a QUIT command to disconnect from the server.\n        '
        self.sendShort(b'QUIT')
from twisted.mail._except import InsecureAuthenticationDisallowed, LineTooLong, ServerErrorResponse, TLSError, TLSNotSupportedError
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
__all__ = ['IMailbox', 'IServerFactory', 'POP3Error', 'POP3ClientError', 'InsecureAuthenticationDisallowed', 'ServerErrorResponse', 'LineTooLong', 'TLSError', 'TLSNotSupportedError', 'POP3', 'POP3Client', 'AdvancedPOP3Client', 'APOPCredentials', 'Mailbox']