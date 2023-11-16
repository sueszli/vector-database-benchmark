"""
A POP3 client protocol implementation.

Don't use this module directly.  Use twisted.mail.pop3 instead.

@author: Jp Calderone
"""
import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import InsecureAuthenticationDisallowed, LineTooLong, ServerErrorResponse, TLSError, TLSNotSupportedError
from twisted.protocols import basic, policies
from twisted.python import log
OK = b'+OK'
ERR = b'-ERR'

class _ListSetter:
    """
    A utility class to construct a list from a multi-line response accounting
    for deleted messages.

    POP3 responses sometimes occur in the form of a list of lines containing
    two pieces of data, a message index and a value of some sort.  When a
    message is deleted, it is omitted from these responses.  The L{setitem}
    method of this class is meant to be called with these two values.  In the
    cases where indices are skipped, it takes care of padding out the missing
    values with L{None}.

    @ivar L: See L{__init__}
    """

    def __init__(self, L):
        if False:
            for i in range(10):
                print('nop')
        '\n        @type L: L{list} of L{object}\n        @param L: The list being constructed.  An empty list should be\n            passed in.\n        '
        self.L = L

    def setitem(self, itemAndValue):
        if False:
            return 10
        '\n        Add the value at the specified position, padding out missing entries.\n\n        @type itemAndValue: C{tuple}\n        @param itemAndValue: A tuple of (item, value).  The I{item} is the 0-based\n        index in the list at which the value should be placed.  The value is\n        is an L{object} to put in the list.\n        '
        (item, value) = itemAndValue
        diff = item - len(self.L) + 1
        if diff > 0:
            self.L.extend([None] * diff)
        self.L[item] = value

def _statXform(line):
    if False:
        return 10
    '\n    Parse the response to a STAT command.\n\n    @type line: L{bytes}\n    @param line: The response from the server to a STAT command minus the\n        status indicator.\n\n    @rtype: 2-L{tuple} of (0) L{int}, (1) L{int}\n    @return: The number of messages in the mailbox and the size of the mailbox.\n    '
    (numMsgs, totalSize) = line.split(None, 1)
    return (int(numMsgs), int(totalSize))

def _listXform(line):
    if False:
        return 10
    '\n    Parse a line of the response to a LIST command.\n\n    The line from the LIST response consists of a 1-based message number\n    followed by a size.\n\n    @type line: L{bytes}\n    @param line: A non-initial line from the multi-line response to a LIST\n        command.\n\n    @rtype: 2-L{tuple} of (0) L{int}, (1) L{int}\n    @return: The 0-based index of the message and the size of the message.\n    '
    (index, size) = line.split(None, 1)
    return (int(index) - 1, int(size))

def _uidXform(line):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse a line of the response to a UIDL command.\n\n    The line from the UIDL response consists of a 1-based message number\n    followed by a unique id.\n\n    @type line: L{bytes}\n    @param line: A non-initial line from the multi-line response to a UIDL\n        command.\n\n    @rtype: 2-L{tuple} of (0) L{int}, (1) L{bytes}\n    @return: The 0-based index of the message and the unique identifier\n        for the message.\n    '
    (index, uid) = line.split(None, 1)
    return (int(index) - 1, uid)

def _codeStatusSplit(line):
    if False:
        print('Hello World!')
    '\n    Parse the first line of a multi-line server response.\n\n    @type line: L{bytes}\n    @param line: The first line of a multi-line server response.\n\n    @rtype: 2-tuple of (0) L{bytes}, (1) L{bytes}\n    @return: The status indicator and the rest of the server response.\n    '
    parts = line.split(b' ', 1)
    if len(parts) == 1:
        return (parts[0], b'')
    return parts

def _dotUnquoter(line):
    if False:
        i = 10
        return i + 15
    "\n    Remove a byte-stuffed termination character at the beginning of a line if\n    present.\n\n    When the termination character (C{'.'}) appears at the beginning of a line,\n    the server byte-stuffs it by adding another termination character to\n    avoid confusion with the terminating sequence (C{'.\\r\\n'}).\n\n    @type line: L{bytes}\n    @param line: A received line.\n\n    @rtype: L{bytes}\n    @return: The line without the byte-stuffed termination character at the\n        beginning if it was present. Otherwise, the line unchanged.\n    "
    if line.startswith(b'..'):
        return line[1:]
    return line

class POP3Client(basic.LineOnlyReceiver, policies.TimeoutMixin):
    """
    A POP3 client protocol.

    Instances of this class provide a convenient, efficient API for
    retrieving and deleting messages from a POP3 server.

    This API provides a pipelining interface but POP3 pipelining
    on the network is not yet supported.

    @type startedTLS: L{bool}
    @ivar startedTLS: An indication of whether TLS has been negotiated
        successfully.

    @type allowInsecureLogin: L{bool}
    @ivar allowInsecureLogin: An indication of whether plaintext login should
        be allowed when the server offers no authentication challenge and the
        transport does not offer any protection via encryption.

    @type serverChallenge: L{bytes} or L{None}
    @ivar serverChallenge: The challenge received in the server greeting.

    @type timeout: L{int}
    @ivar timeout: The number of seconds to wait on a response from the server
        before timing out a connection.  If the number is <= 0, no timeout
        checking will be performed.

    @type _capCache: L{None} or L{dict} mapping L{bytes}
        to L{list} of L{bytes} and/or L{bytes} to L{None}
    @ivar _capCache: The cached server capabilities.  Capabilities are not
        allowed to change during the session (except when TLS is negotiated),
        so the first response to a capabilities command can be used for
        later lookups.

    @type _challengeMagicRe: L{Pattern <re.Pattern.search>}
    @ivar _challengeMagicRe: A regular expression which matches the
        challenge in the server greeting.

    @type _blockedQueue: L{None} or L{list} of 3-L{tuple}
        of (0) L{Deferred <defer.Deferred>}, (1) callable which results
        in a L{Deferred <defer.Deferred>}, (2) L{tuple}
    @ivar _blockedQueue: A list of blocked commands.  While a command is
        awaiting a response from the server, other commands are blocked.  When
        no command is outstanding, C{_blockedQueue} is set to L{None}.
        Otherwise, it contains a list of information about blocked commands.
        Each list entry provides the following information about a blocked
        command: the deferred that should be called when the response to the
        command is received, the function that sends the command, and the
        arguments to the function.

    @type _waiting: L{Deferred <defer.Deferred>} or
        L{None}
    @ivar _waiting: A deferred which fires when the response to the
        outstanding command is received from the server.

    @type _timedOut: L{bool}
    @ivar _timedOut: An indication of whether the connection was dropped
        because of a timeout.

    @type _greetingError: L{bytes} or L{None}
    @ivar _greetingError: The server greeting minus the status indicator, when
        the connection was dropped because of an error in the server greeting.
        Otherwise, L{None}.

    @type state: L{bytes}
    @ivar state: The state which indicates what type of response is expected
        from the server.  Valid states are: 'WELCOME', 'WAITING', 'SHORT',
        'LONG_INITIAL', 'LONG'.

    @type _xform: L{None} or callable that takes L{bytes}
        and returns L{object}
    @ivar _xform: The transform function which is used to convert each
        line of a multi-line response into usable values for use by the
        consumer function.  If L{None}, each line of the multi-line response
        is sent directly to the consumer function.

    @type _consumer: callable that takes L{object}
    @ivar _consumer: The consumer function which is used to store the
        values derived by the transform function from each line of a
        multi-line response into a list.
    """
    startedTLS = False
    allowInsecureLogin = False
    timeout = 0
    serverChallenge = None
    _capCache = None
    _challengeMagicRe = re.compile(b'(<[^>]+>)')
    _blockedQueue = None
    _waiting = None
    _timedOut = False
    _greetingError = None

    def _blocked(self, f, *a):
        if False:
            for i in range(10):
                print('nop')
        '\n        Block a command, if necessary.\n\n        If commands are being blocked, append information about the function\n        which sends the command to a list and return a deferred that will be\n        chained with the return value of the function when it eventually runs.\n        Otherwise, set up for subsequent commands to be blocked and return\n        L{None}.\n\n        @type f: callable\n        @param f: A function which sends a command.\n\n        @type a: L{tuple}\n        @param a: Arguments to the function.\n\n        @rtype: L{None} or L{Deferred <defer.Deferred>}\n        @return: L{None} if the command can run immediately.  Otherwise,\n            a deferred that will eventually trigger with the return value of\n            the function.\n        '
        if self._blockedQueue is not None:
            d = defer.Deferred()
            self._blockedQueue.append((d, f, a))
            return d
        self._blockedQueue = []
        return None

    def _unblock(self):
        if False:
            return 10
        '\n        Send the next blocked command.\n\n        If there are no more commands in the blocked queue, set up for the next\n        command to be sent immediately.\n        '
        if self._blockedQueue == []:
            self._blockedQueue = None
        elif self._blockedQueue is not None:
            _blockedQueue = self._blockedQueue
            self._blockedQueue = None
            (d, f, a) = _blockedQueue.pop(0)
            d2 = f(*a)
            d2.chainDeferred(d)
            self._blockedQueue.extend(_blockedQueue)

    def sendShort(self, cmd, args):
        if False:
            i = 10
            return i + 15
        '\n        Send a POP3 command to which a short response is expected.\n\n        Block all further commands from being sent until the response is\n        received.  Transition the state to SHORT.\n\n        @type cmd: L{bytes}\n        @param cmd: A POP3 command.\n\n        @type args: L{bytes}\n        @param args: The command arguments.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the entire response is received.\n            On an OK response, it returns the response from the server minus\n            the status indicator.  On an ERR response, it issues a server\n            error response failure with the response from the server minus the\n            status indicator.\n        '
        d = self._blocked(self.sendShort, cmd, args)
        if d is not None:
            return d
        if args:
            self.sendLine(cmd + b' ' + args)
        else:
            self.sendLine(cmd)
        self.state = 'SHORT'
        self._waiting = defer.Deferred()
        return self._waiting

    def sendLong(self, cmd, args, consumer, xform):
        if False:
            while True:
                i = 10
        '\n        Send a POP3 command to which a multi-line response is expected.\n\n        Block all further commands from being sent until the entire response is\n        received.  Transition the state to LONG_INITIAL.\n\n        @type cmd: L{bytes}\n        @param cmd: A POP3 command.\n\n        @type args: L{bytes}\n        @param args: The command arguments.\n\n        @type consumer: callable that takes L{object}\n        @param consumer: A consumer function which should be used to put\n            the values derived by a transform function from each line of the\n            multi-line response into a list.\n\n        @type xform: L{None} or callable that takes\n            L{bytes} and returns L{object}\n        @param xform: A transform function which should be used to transform\n            each line of the multi-line response into usable values for use by\n            a consumer function.  If L{None}, each line of the multi-line\n            response should be sent directly to the consumer function.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            callable that takes L{object} and fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the entire response is received.\n            On an OK response, it returns the consumer function.  On an ERR\n            response, it issues a server error response failure with the\n            response from the server minus the status indicator and the\n            consumer function.\n        '
        d = self._blocked(self.sendLong, cmd, args, consumer, xform)
        if d is not None:
            return d
        if args:
            self.sendLine(cmd + b' ' + args)
        else:
            self.sendLine(cmd)
        self.state = 'LONG_INITIAL'
        self._xform = xform
        self._consumer = consumer
        self._waiting = defer.Deferred()
        return self._waiting

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait for a greeting from the server after the connection has been made.\n\n        Start the connection in the WELCOME state.\n        '
        if self.timeout > 0:
            self.setTimeout(self.timeout)
        self.state = 'WELCOME'
        self._blockedQueue = []

    def timeoutConnection(self):
        if False:
            return 10
        '\n        Drop the connection when the server does not respond in time.\n        '
        self._timedOut = True
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        '\n        Clean up when the connection has been lost.\n\n        When the loss of connection was initiated by the client due to a\n        timeout, the L{_timedOut} flag will be set.  When it was initiated by\n        the client due to an error in the server greeting, L{_greetingError}\n        will be set to the server response minus the status indicator.\n\n        @type reason: L{Failure <twisted.python.failure.Failure>}\n        @param reason: The reason the connection was terminated.\n        '
        if self.timeout > 0:
            self.setTimeout(None)
        if self._timedOut:
            reason = error.TimeoutError()
        elif self._greetingError:
            reason = ServerErrorResponse(self._greetingError)
        d = []
        if self._waiting is not None:
            d.append(self._waiting)
            self._waiting = None
        if self._blockedQueue is not None:
            d.extend([deferred for (deferred, f, a) in self._blockedQueue])
            self._blockedQueue = None
        for w in d:
            w.errback(reason)

    def lineReceived(self, line):
        if False:
            print('Hello World!')
        '\n        Pass a received line to a state machine function and\n        transition to the next state.\n\n        @type line: L{bytes}\n        @param line: A received line.\n        '
        if self.timeout > 0:
            self.resetTimeout()
        state = self.state
        self.state = None
        state = getattr(self, 'state_' + state)(line) or state
        if self.state is None:
            self.state = state

    def lineLengthExceeded(self, buffer):
        if False:
            return 10
        '\n        Drop the connection when a server response exceeds the maximum line\n        length (L{LineOnlyReceiver.MAX_LENGTH}).\n\n        @type buffer: L{bytes}\n        @param buffer: A received line which exceeds the maximum line length.\n        '
        if self._waiting is not None:
            (waiting, self._waiting) = (self._waiting, None)
            waiting.errback(LineTooLong())
        self.transport.loseConnection()

    def state_WELCOME(self, line):
        if False:
            print('Hello World!')
        '\n        Handle server responses for the WELCOME state in which the server\n        greeting is expected.\n\n        WELCOME is the first state.  The server should send one line of text\n        with a greeting and possibly an APOP challenge.  Transition the state\n        to WAITING.\n\n        @type line: L{bytes}\n        @param line: A line received from the server.\n\n        @rtype: L{bytes}\n        @return: The next state.\n        '
        (code, status) = _codeStatusSplit(line)
        if code != OK:
            self._greetingError = status
            self.transport.loseConnection()
        else:
            m = self._challengeMagicRe.search(status)
            if m is not None:
                self.serverChallenge = m.group(1)
            self.serverGreeting(status)
        self._unblock()
        return 'WAITING'

    def state_WAITING(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Log an error for server responses received in the WAITING state during\n        which the server is not expected to send anything.\n\n        @type line: L{bytes}\n        @param line: A line received from the server.\n        '
        log.msg('Illegal line from server: ' + repr(line))

    def state_SHORT(self, line):
        if False:
            return 10
        '\n        Handle server responses for the SHORT state in which the server is\n        expected to send a single line response.\n\n        Parse the response and fire the deferred which is waiting on receipt of\n        a complete response.  Transition the state back to WAITING.\n\n        @type line: L{bytes}\n        @param line: A line received from the server.\n\n        @rtype: L{bytes}\n        @return: The next state.\n        '
        (deferred, self._waiting) = (self._waiting, None)
        self._unblock()
        (code, status) = _codeStatusSplit(line)
        if code == OK:
            deferred.callback(status)
        else:
            deferred.errback(ServerErrorResponse(status))
        return 'WAITING'

    def state_LONG_INITIAL(self, line):
        if False:
            return 10
        '\n        Handle server responses for the LONG_INITIAL state in which the server\n        is expected to send the first line of a multi-line response.\n\n        Parse the response.  On an OK response, transition the state to\n        LONG.  On an ERR response, cleanup and transition the state to\n        WAITING.\n\n        @type line: L{bytes}\n        @param line: A line received from the server.\n\n        @rtype: L{bytes}\n        @return: The next state.\n        '
        (code, status) = _codeStatusSplit(line)
        if code == OK:
            return 'LONG'
        consumer = self._consumer
        deferred = self._waiting
        self._consumer = self._waiting = self._xform = None
        self._unblock()
        deferred.errback(ServerErrorResponse(status, consumer))
        return 'WAITING'

    def state_LONG(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle server responses for the LONG state in which the server is\n        expected to send a non-initial line of a multi-line response.\n\n        On receipt of the last line of the response, clean up, fire the\n        deferred which is waiting on receipt of a complete response, and\n        transition the state to WAITING. Otherwise, pass the line to the\n        transform function, if provided, and then the consumer function.\n\n        @type line: L{bytes}\n        @param line: A line received from the server.\n\n        @rtype: L{bytes}\n        @return: The next state.\n        '
        if line == b'.':
            consumer = self._consumer
            deferred = self._waiting
            self._consumer = self._waiting = self._xform = None
            self._unblock()
            deferred.callback(consumer)
            return 'WAITING'
        else:
            if self._xform is not None:
                self._consumer(self._xform(line))
            else:
                self._consumer(line)
            return 'LONG'

    def serverGreeting(self, greeting):
        if False:
            while True:
                i = 10
        '\n        Handle the server greeting.\n\n        @type greeting: L{bytes}\n        @param greeting: The server greeting minus the status indicator.\n            For servers implementing APOP authentication, this will contain a\n            challenge string.\n        '

    def startTLS(self, contextFactory=None):
        if False:
            print('Hello World!')
        "\n        Switch to encrypted communication using TLS.\n\n        The first step of switching to encrypted communication is obtaining\n        the server's capabilities.  When that is complete, the L{_startTLS}\n        callback function continues the switching process.\n\n        @type contextFactory: L{None} or\n            L{ClientContextFactory <twisted.internet.ssl.ClientContextFactory>}\n        @param contextFactory: The context factory with which to negotiate TLS.\n            If not provided, try to create a new one.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully results in\n            L{dict} mapping L{bytes} to L{list} of L{bytes} and/or L{bytes} to\n            L{None} or fails with L{TLSError}\n        @return: A deferred which fires when the transport has been\n            secured according to the given context factory with the server\n            capabilities, or which fails with a TLS error if the transport\n            cannot be secured.\n        "
        tls = interfaces.ITLSTransport(self.transport, None)
        if tls is None:
            return defer.fail(TLSError('POP3Client transport does not implement interfaces.ITLSTransport'))
        if contextFactory is None:
            contextFactory = self._getContextFactory()
        if contextFactory is None:
            return defer.fail(TLSError('POP3Client requires a TLS context to initiate the STLS handshake'))
        d = self.capabilities()
        d.addCallback(self._startTLS, contextFactory, tls)
        return d

    def _startTLS(self, caps, contextFactory, tls):
        if False:
            print('Hello World!')
        '\n        Continue the process of switching to encrypted communication.\n\n        This callback function runs after the server capabilities are received.\n\n        The next step is sending the server an STLS command to request a\n        switch to encrypted communication.  When an OK response is received,\n        the L{_startedTLS} callback function completes the switch to encrypted\n        communication. Then, the new server capabilities are requested.\n\n        @type caps: L{dict} mapping L{bytes} to L{list} of L{bytes} and/or\n            L{bytes} to L{None}\n        @param caps: The server capabilities.\n\n        @type contextFactory: L{ClientContextFactory\n            <twisted.internet.ssl.ClientContextFactory>}\n        @param contextFactory: A context factory with which to negotiate TLS.\n\n        @type tls: L{ITLSTransport <interfaces.ITLSTransport>}\n        @param tls: A TCP transport that supports switching to TLS midstream.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully triggers with\n            L{dict} mapping L{bytes} to L{list} of L{bytes} and/or L{bytes} to\n            L{None} or fails with L{TLSNotSupportedError}\n        @return: A deferred which successfully fires when the response from\n            the server to the request to start TLS has been received and the\n            new server capabilities have been received or fails when the server\n            does not support TLS.\n        '
        assert not self.startedTLS, 'Client and Server are currently communicating via TLS'
        if b'STLS' not in caps:
            return defer.fail(TLSNotSupportedError('Server does not support secure communication via TLS / SSL'))
        d = self.sendShort(b'STLS', None)
        d.addCallback(self._startedTLS, contextFactory, tls)
        d.addCallback(lambda _: self.capabilities())
        return d

    def _startedTLS(self, result, context, tls):
        if False:
            print('Hello World!')
        '\n        Complete the process of switching to encrypted communication.\n\n        This callback function runs after the response to the STLS command has\n        been received.\n\n        The final steps are discarding the cached capabilities and initiating\n        TLS negotiation on the transport.\n\n        @type result: L{dict} mapping L{bytes} to L{list} of L{bytes} and/or\n            L{bytes} to L{None}\n        @param result: The server capabilities.\n\n        @type context: L{ClientContextFactory\n            <twisted.internet.ssl.ClientContextFactory>}\n        @param context: A context factory with which to negotiate TLS.\n\n        @type tls: L{ITLSTransport <interfaces.ITLSTransport>}\n        @param tls: A TCP transport that supports switching to TLS midstream.\n\n        @rtype: L{dict} mapping L{bytes} to L{list} of L{bytes} and/or L{bytes}\n            to L{None}\n        @return: The server capabilities.\n        '
        self.transport = tls
        self.transport.startTLS(context)
        self._capCache = None
        self.startedTLS = True
        return result

    def _getContextFactory(self):
        if False:
            return 10
        '\n        Get a context factory with which to negotiate TLS.\n\n        @rtype: L{None} or\n            L{ClientContextFactory <twisted.internet.ssl.ClientContextFactory>}\n        @return: A context factory or L{None} if TLS is not supported on the\n            client.\n        '
        try:
            from twisted.internet import ssl
        except ImportError:
            return None
        else:
            context = ssl.ClientContextFactory()
            context.method = ssl.SSL.TLSv1_2_METHOD
            return context

    def login(self, username, password):
        if False:
            while True:
                i = 10
        "\n        Log in to the server.\n\n        If APOP is available it will be used.  Otherwise, if TLS is\n        available, an encrypted session will be started and plaintext\n        login will proceed.  Otherwise, if L{allowInsecureLogin} is set,\n        insecure plaintext login will proceed.  Otherwise,\n        L{InsecureAuthenticationDisallowed} will be raised.\n\n        The first step of logging into the server is obtaining the server's\n        capabilities.  When that is complete, the L{_login} callback function\n        continues the login process.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @type password: L{bytes}\n        @param password: The password with which to log in.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes}\n        @return: A deferred which fires when the login process is complete.\n            On a successful login, it returns the server's response minus the\n            status indicator.\n        "
        d = self.capabilities()
        d.addCallback(self._login, username, password)
        return d

    def _login(self, caps, username, password):
        if False:
            while True:
                i = 10
        "\n        Continue the process of logging in to the server.\n\n        This callback function runs after the server capabilities are received.\n\n        If the server provided a challenge in the greeting, proceed with an\n        APOP login.  Otherwise, if the server and the transport support\n        encrypted communication, try to switch to TLS and then complete\n        the login process with the L{_loginTLS} callback function.  Otherwise,\n        if insecure authentication is allowed, do a plaintext login.\n        Otherwise, fail with an L{InsecureAuthenticationDisallowed} error.\n\n        @type caps: L{dict} mapping L{bytes} to L{list} of L{bytes} and/or\n            L{bytes} to L{None}\n        @param caps: The server capabilities.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @type password: L{bytes}\n        @param password: The password with which to log in.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes}\n        @return: A deferred which fires when the login process is complete.\n            On a successful login, it returns the server's response minus the\n            status indicator.\n        "
        if self.serverChallenge is not None:
            return self._apop(username, password, self.serverChallenge)
        tryTLS = b'STLS' in caps
        tlsableTransport = interfaces.ITLSTransport(self.transport, None) is not None
        nontlsTransport = interfaces.ISSLTransport(self.transport, None) is None
        if not self.startedTLS and tryTLS and tlsableTransport and nontlsTransport:
            d = self.startTLS()
            d.addCallback(self._loginTLS, username, password)
            return d
        elif self.startedTLS or not nontlsTransport or self.allowInsecureLogin:
            return self._plaintext(username, password)
        else:
            return defer.fail(InsecureAuthenticationDisallowed())

    def _loginTLS(self, res, username, password):
        if False:
            return 10
        "\n        Do a plaintext login over an encrypted transport.\n\n        This callback function runs after the transport switches to encrypted\n        communication.\n\n        @type res: L{dict} mapping L{bytes} to L{list} of L{bytes} and/or\n            L{bytes} to L{None}\n        @param res: The server capabilities.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @type password: L{bytes}\n        @param password: The password with which to log in.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server accepts the username\n            and password or fails when the server rejects either.  On a\n            successful login, it returns the server's response minus the\n            status indicator.\n        "
        return self._plaintext(username, password)

    def _plaintext(self, username, password):
        if False:
            print('Hello World!')
        "\n        Perform a plaintext login.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @type password: L{bytes}\n        @param password: The password with which to log in.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server accepts the username\n            and password or fails when the server rejects either.  On a\n            successful login, it returns the server's response minus the\n            status indicator.\n        "
        return self.user(username).addCallback(lambda r: self.password(password))

    def _apop(self, username, password, challenge):
        if False:
            print('Hello World!')
        '\n        Perform an APOP login.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @type password: L{bytes}\n        @param password: The password with which to log in.\n\n        @type challenge: L{bytes}\n        @param challenge: A challenge string.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On a successful login, it returns the server response minus\n            the status indicator.\n        '
        digest = md5(challenge + password).hexdigest().encode('ascii')
        return self.apop(username, digest)

    def apop(self, username, digest):
        if False:
            print('Hello World!')
        '\n        Send an APOP command to perform authenticated login.\n\n        This should be used in special circumstances only, when it is\n        known that the server supports APOP authentication, and APOP\n        authentication is absolutely required.  For the common case,\n        use L{login} instead.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @type digest: L{bytes}\n        @param digest: The challenge response to authenticate with.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'APOP', username + b' ' + digest)

    def user(self, username):
        if False:
            i = 10
            return i + 15
        '\n        Send a USER command to perform the first half of plaintext login.\n\n        Unless this is absolutely required, use the L{login} method instead.\n\n        @type username: L{bytes}\n        @param username: The username with which to log in.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'USER', username)

    def password(self, password):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a PASS command to perform the second half of plaintext login.\n\n        Unless this is absolutely required, use the L{login} method instead.\n\n        @type password: L{bytes}\n        @param password: The plaintext password with which to authenticate.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'PASS', password)

    def delete(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a DELE command to delete a message from the server.\n\n        @type index: L{int}\n        @param index: The 0-based index of the message to delete.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'DELE', b'%d' % (index + 1,))

    def _consumeOrSetItem(self, cmd, args, consumer, xform):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a command to which a long response is expected and process the\n        multi-line response into a list accounting for deleted messages.\n\n        @type cmd: L{bytes}\n        @param cmd: A POP3 command to which a long response is expected.\n\n        @type args: L{bytes}\n        @param args: The command arguments.\n\n        @type consumer: L{None} or callable that takes\n            L{object}\n        @param consumer: L{None} or a function that consumes the output from\n            the transform function.\n\n        @type xform: L{None}, callable that takes\n            L{bytes} and returns 2-L{tuple} of (0) L{int}, (1) L{object},\n            or callable that takes L{bytes} and returns L{object}\n        @param xform: A function that parses a line from a multi-line response\n            and transforms the values into usable form for input to the\n            consumer function.  If no consumer function is specified, the\n            output must be a message index and corresponding value.  If no\n            transform function is specified, the line is used as is.\n\n        @rtype: L{Deferred <defer.Deferred>} which fires with L{list} of\n            L{object} or callable that takes L{list} of L{object}\n        @return: A deferred which fires when the entire response has been\n            received.  When a consumer is not provided, the return value is a\n            list of the value for each message or L{None} for deleted messages.\n            Otherwise, it returns the consumer itself.\n        '
        if consumer is None:
            L = []
            consumer = _ListSetter(L).setitem
            return self.sendLong(cmd, args, consumer, xform).addCallback(lambda r: L)
        return self.sendLong(cmd, args, consumer, xform)

    def _consumeOrAppend(self, cmd, args, consumer, xform):
        if False:
            while True:
                i = 10
        '\n        Send a command to which a long response is expected and process the\n        multi-line response into a list.\n\n        @type cmd: L{bytes}\n        @param cmd: A POP3 command which expects a long response.\n\n        @type args: L{bytes}\n        @param args: The command arguments.\n\n        @type consumer: L{None} or callable that takes\n            L{object}\n        @param consumer: L{None} or a function that consumes the output from the\n            transform function.\n\n        @type xform: L{None} or callable that takes\n            L{bytes} and returns L{object}\n        @param xform: A function that transforms a line from a multi-line\n            response into usable form for input to the consumer function.  If\n            no transform function is specified, the line is used as is.\n\n        @rtype: L{Deferred <defer.Deferred>} which fires with L{list} of\n            2-L{tuple} of (0) L{int}, (1) L{object} or callable that\n            takes 2-L{tuple} of (0) L{int}, (1) L{object}\n        @return: A deferred which fires when the entire response has been\n            received.  When a consumer is not provided, the return value is a\n            list of the transformed lines.  Otherwise, it returns the consumer\n            itself.\n        '
        if consumer is None:
            L = []
            consumer = L.append
            return self.sendLong(cmd, args, consumer, xform).addCallback(lambda r: L)
        return self.sendLong(cmd, args, consumer, xform)

    def capabilities(self, useCache=True):
        if False:
            print('Hello World!')
        "\n        Send a CAPA command to retrieve the capabilities supported by\n        the server.\n\n        Not all servers support this command.  If the server does not\n        support this, it is treated as though it returned a successful\n        response listing no capabilities.  At some future time, this may be\n        changed to instead seek out information about a server's\n        capabilities in some other fashion (only if it proves useful to do\n        so, and only if there are servers still in use which do not support\n        CAPA but which do support POP3 extensions that are useful).\n\n        @type useCache: L{bool}\n        @param useCache: A flag that determines whether previously retrieved\n            results should be used if available.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully results in\n            L{dict} mapping L{bytes} to L{list} of L{bytes} and/or L{bytes} to\n            L{None}\n        @return: A deferred which fires with a mapping of capability name to\n        parameters.  For example::\n\n            C: CAPA\n            S: +OK Capability list follows\n            S: TOP\n            S: USER\n            S: SASL CRAM-MD5 KERBEROS_V4\n            S: RESP-CODES\n            S: LOGIN-DELAY 900\n            S: PIPELINING\n            S: EXPIRE 60\n            S: UIDL\n            S: IMPLEMENTATION Shlemazle-Plotz-v302\n            S: .\n\n        will be lead to a result of::\n\n            | {'TOP': None,\n            |  'USER': None,\n            |  'SASL': ['CRAM-MD5', 'KERBEROS_V4'],\n            |  'RESP-CODES': None,\n            |  'LOGIN-DELAY': ['900'],\n            |  'PIPELINING': None,\n            |  'EXPIRE': ['60'],\n            |  'UIDL': None,\n            |  'IMPLEMENTATION': ['Shlemazle-Plotz-v302']}\n        "
        if useCache and self._capCache is not None:
            return defer.succeed(self._capCache)
        cache = {}

        def consume(line):
            if False:
                print('Hello World!')
            tmp = line.split()
            if len(tmp) == 1:
                cache[tmp[0]] = None
            elif len(tmp) > 1:
                cache[tmp[0]] = tmp[1:]

        def capaNotSupported(err):
            if False:
                return 10
            err.trap(ServerErrorResponse)
            return None

        def gotCapabilities(result):
            if False:
                while True:
                    i = 10
            self._capCache = cache
            return cache
        d = self._consumeOrAppend(b'CAPA', None, consume, None)
        d.addErrback(capaNotSupported).addCallback(gotCapabilities)
        return d

    def noop(self):
        if False:
            print('Hello World!')
        '\n        Send a NOOP command asking the server to do nothing but respond.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'NOOP', None)

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Send a RSET command to unmark any messages that have been flagged\n        for deletion on the server.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'RSET', None)

    def retrieve(self, index, consumer=None, lines=None):
        if False:
            print('Hello World!')
        '\n        Send a RETR or TOP command to retrieve all or part of a message from\n        the server.\n\n        @type index: L{int}\n        @param index: A 0-based message index.\n\n        @type consumer: L{None} or callable that takes\n            L{bytes}\n        @param consumer: A function which consumes each transformed line from a\n            multi-line response as it is received.\n\n        @type lines: L{None} or L{int}\n        @param lines: If specified, the number of lines of the message to be\n            retrieved.  Otherwise, the entire message is retrieved.\n\n        @rtype: L{Deferred <defer.Deferred>} which fires with L{list} of\n            L{bytes}, or callable that takes 2-L{tuple} of (0) L{int},\n            (1) L{object}\n        @return: A deferred which fires when the entire response has been\n            received.  When a consumer is not provided, the return value is a\n            list of the transformed lines.  Otherwise, it returns the consumer\n            itself.\n        '
        idx = b'%d' % (index + 1,)
        if lines is None:
            return self._consumeOrAppend(b'RETR', idx, consumer, _dotUnquoter)
        return self._consumeOrAppend(b'TOP', b'%b %d' % (idx, lines), consumer, _dotUnquoter)

    def stat(self):
        if False:
            print('Hello World!')
        '\n        Send a STAT command to get information about the size of the mailbox.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            a 2-tuple of (0) L{int}, (1) L{int} or fails with\n            L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the number of\n            messages in the mailbox and the size of the mailbox in octets.\n            On an ERR response, the deferred fails with a server error\n            response failure.\n        '
        return self.sendShort(b'STAT', None).addCallback(_statXform)

    def listSize(self, consumer=None):
        if False:
            return 10
        '\n        Send a LIST command to retrieve the sizes of all messages on the\n        server.\n\n        @type consumer: L{None} or callable that takes\n            2-L{tuple} of (0) L{int}, (1) L{int}\n        @param consumer: A function which consumes the 0-based message index\n            and message size derived from the server response.\n\n        @rtype: L{Deferred <defer.Deferred>} which fires L{list} of L{int} or\n            callable that takes 2-L{tuple} of (0) L{int}, (1) L{int}\n        @return: A deferred which fires when the entire response has been\n            received.  When a consumer is not provided, the return value is a\n            list of message sizes.  Otherwise, it returns the consumer itself.\n        '
        return self._consumeOrSetItem(b'LIST', None, consumer, _listXform)

    def listUID(self, consumer=None):
        if False:
            print('Hello World!')
        '\n        Send a UIDL command to retrieve the UIDs of all messages on the server.\n\n        @type consumer: L{None} or callable that takes\n            2-L{tuple} of (0) L{int}, (1) L{bytes}\n        @param consumer: A function which consumes the 0-based message index\n            and UID derived from the server response.\n\n        @rtype: L{Deferred <defer.Deferred>} which fires with L{list} of\n            L{object} or callable that takes 2-L{tuple} of (0) L{int},\n            (1) L{bytes}\n        @return: A deferred which fires when the entire response has been\n            received.  When a consumer is not provided, the return value is a\n            list of message sizes.  Otherwise, it returns the consumer itself.\n        '
        return self._consumeOrSetItem(b'UIDL', None, consumer, _uidXform)

    def quit(self):
        if False:
            print('Hello World!')
        '\n        Send a QUIT command to disconnect from the server.\n\n        @rtype: L{Deferred <defer.Deferred>} which successfully fires with\n            L{bytes} or fails with L{ServerErrorResponse}\n        @return: A deferred which fires when the server response is received.\n            On an OK response, the deferred succeeds with the server\n            response minus the status indicator.  On an ERR response, the\n            deferred fails with a server error response failure.\n        '
        return self.sendShort(b'QUIT', None)
__all__: List[str] = []