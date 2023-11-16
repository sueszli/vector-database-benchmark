"""
An IMAP4 protocol implementation

@author: Jp Calderone

To do::
  Suspend idle timeout while server is processing
  Use an async message parser instead of buffering in memory
  Figure out a way to not queue multi-message client requests (Flow? A simple callback?)
  Clarify some API docs (Query, etc)
  Make APPEND recognize (again) non-existent mailboxes before accepting the literal
"""
import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import CramMD5ClientAuthenticator, LOGINAuthenticator, LOGINCredentials, PLAINAuthenticator, PLAINCredentials
from twisted.mail._except import IllegalClientResponse, IllegalIdentifierError, IllegalMailboxEncoding, IllegalOperation, IllegalQueryError, IllegalServerResponse, IMAP4Exception, MailboxCollision, MailboxException, MismatchedNesting, MismatchedQuoting, NegativeResponse, NoSuchMailbox, NoSupportedAuthentication, ReadOnlyMailbox, UnhandledResponse
from twisted.mail.interfaces import IAccountIMAP as IAccount, IClientAuthentication, ICloseableMailboxIMAP as ICloseableMailbox, IMailboxIMAP as IMailbox, IMailboxIMAPInfo as IMailboxInfo, IMailboxIMAPListener as IMailboxListener, IMessageIMAP as IMessage, IMessageIMAPCopier as IMessageCopier, IMessageIMAPFile as IMessageFile, IMessageIMAPPart as IMessagePart, INamespacePresenter, ISearchableIMAPMailbox as ISearchableMailbox
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import _get_async_param, _matchingString, iterbytes, nativeString, networkString
_MONTH_NAMES = dict(zip(range(1, 13), 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()))

def _swap(this, that, ifIs):
    if False:
        print('Hello World!')
    '\n    Swap C{this} with C{that} if C{this} is C{ifIs}.\n\n    @param this: The object that may be replaced.\n\n    @param that: The object that may replace C{this}.\n\n    @param ifIs: An object whose identity will be compared to\n        C{this}.\n    '
    return that if this is ifIs else this

def _swapAllPairs(of, that, ifIs):
    if False:
        print('Hello World!')
    '\n    Swap each element in each pair in C{of} with C{that} it is\n    C{ifIs}.\n\n    @param of: A list of 2-L{tuple}s, whose members may be the object\n        C{that}\n    @type of: L{list} of 2-L{tuple}s\n\n    @param ifIs: An object whose identity will be compared to members\n        of each pair in C{of}\n\n    @return: A L{list} of 2-L{tuple}s with all occurences of C{ifIs}\n        replaced with C{that}\n    '
    return [(_swap(first, that, ifIs), _swap(second, that, ifIs)) for (first, second) in of]

class MessageSet:
    """
    A set of message identifiers usable by both L{IMAP4Client} and
    L{IMAP4Server} via L{IMailboxIMAP.store} and
    L{IMailboxIMAP.fetch}.

    These identifiers can be either message sequence numbers or unique
    identifiers.  See Section 2.3.1, "Message Numbers", RFC 3501.

    This represents the C{sequence-set} described in Section 9,
    "Formal Syntax" of RFC 3501:

        - A L{MessageSet} can describe a single identifier, e.g.
          C{MessageSet(1)}

        - A L{MessageSet} can describe C{*} via L{None}, e.g.
          C{MessageSet(None)}

        - A L{MessageSet} can describe a range of identifiers, e.g.
          C{MessageSet(1, 2)}.  The range is inclusive and unordered
          (see C{seq-range} in RFC 3501, Section 9), so that
          C{Message(2, 1)} is equivalent to C{MessageSet(1, 2)}, and
          both describe messages 1 and 2.  Ranges can include C{*} by
          specifying L{None}, e.g. C{MessageSet(None, 1)}.  In all
          cases ranges are normalized so that the smallest identifier
          comes first, and L{None} always comes last; C{Message(2, 1)}
          becomes C{MessageSet(1, 2)} and C{MessageSet(None, 1)}
          becomes C{MessageSet(1, None)}

        - A L{MessageSet} can describe a sequence of single
          identifiers and ranges, constructed by addition.
          C{MessageSet(1) + MessageSet(5, 10)} refers the message
          identified by C{1} and the messages identified by C{5}
          through C{10}.

    B{NB: The meaning of * varies, but it always represents the
    largest number in use}.

    B{For servers}: Your L{IMailboxIMAP} provider must set
    L{MessageSet.last} to the highest-valued identifier (unique or
    message sequence) before iterating over it.

    B{For clients}: C{*} consumes ranges smaller than it, e.g.
    C{MessageSet(1, 100) + MessageSet(50, None)} is equivalent to
    C{1:*}.

    @type getnext: Function taking L{int} returning L{int}
    @ivar getnext: A function that returns the next message number,
        used when iterating through the L{MessageSet}.  By default, a
        function returning the next integer is supplied, but as this
        can be rather inefficient for sparse UID iterations, it is
        recommended to supply one when messages are requested by UID.
        The argument is provided as a hint to the implementation and
        may be ignored if it makes sense to do so (eg, if an iterator
        is being used that maintains its own state, it is guaranteed
        that it will not be called out-of-order).
    """
    _empty: List[Any] = []
    _infinity = float('inf')

    def __init__(self, start=_empty, end=_empty):
        if False:
            while True:
                i = 10
        '\n        Create a new MessageSet()\n\n        @type start: Optional L{int}\n        @param start: Start of range, or only message number\n\n        @type end: Optional L{int}\n        @param end: End of range.\n        '
        self._last = self._empty
        self.ranges = []
        self.getnext = lambda x: x + 1
        if start is self._empty:
            return
        if isinstance(start, list):
            self.ranges = start[:]
            self.clean()
        else:
            self.add(start, end)

    @property
    def last(self):
        if False:
            return 10
        '\n        The largest number in use.\n        This is undefined until it has been set by assigning to this property.\n        '
        return self._last

    @last.setter
    def last(self, value):
        if False:
            return 10
        '\n        Replaces all occurrences of "*".  This should be the\n        largest number in use.  Must be set before attempting to\n        use the MessageSet as a container.\n\n        @raises ValueError: if a largest value has already been set.\n        '
        if self._last is not self._empty:
            raise ValueError('last already set')
        self._last = value
        for (i, (low, high)) in enumerate(self.ranges):
            if low is None:
                low = value
            if high is None:
                high = value
            if low > high:
                (low, high) = (high, low)
            self.ranges[i] = (low, high)
        self.clean()

    def add(self, start, end=_empty):
        if False:
            return 10
        '\n        Add another range\n\n        @type start: L{int}\n        @param start: Start of range, or only message number\n\n        @type end: Optional L{int}\n        @param end: End of range.\n        '
        if end is self._empty:
            end = start
        if self._last is not self._empty:
            if start is None:
                start = self.last
            if end is None:
                end = self.last
        (start, end) = sorted([start, end], key=functools.partial(_swap, that=self._infinity, ifIs=None))
        self.ranges.append((start, end))
        self.clean()

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, MessageSet):
            ranges = self.ranges + other.ranges
            return MessageSet(ranges)
        else:
            res = MessageSet(self.ranges)
            if self.last is not self._empty:
                res.last = self.last
            try:
                res.add(*other)
            except TypeError:
                res.add(other)
            return res

    def extend(self, other):
        if False:
            print('Hello World!')
        '\n        Extend our messages with another message or set of messages.\n\n        @param other: The messages to include.\n        @type other: L{MessageSet}, L{tuple} of two L{int}s, or a\n            single L{int}\n        '
        if isinstance(other, MessageSet):
            self.ranges.extend(other.ranges)
            self.clean()
        else:
            try:
                self.add(*other)
            except TypeError:
                self.add(other)
        return self

    def clean(self):
        if False:
            return 10
        '\n        Clean ranges list, combining adjacent ranges\n        '
        ranges = sorted(_swapAllPairs(self.ranges, that=self._infinity, ifIs=None))
        mergedRanges = [(float('-inf'), float('-inf'))]
        for (low, high) in ranges:
            (previousLow, previousHigh) = mergedRanges[-1]
            if previousHigh < low - 1:
                mergedRanges.append((low, high))
                continue
            mergedRanges[-1] = (min(previousLow, low), max(previousHigh, high))
        self.ranges = _swapAllPairs(mergedRanges[1:], that=None, ifIs=self._infinity)

    def _noneInRanges(self):
        if False:
            i = 10
            return i + 15
        '\n        Is there a L{None} in our ranges?\n\n        L{MessageSet.clean} merges overlapping or consecutive ranges.\n        None is represents a value larger than any number.  There are\n        thus two cases:\n\n            1. C{(x, *) + (y, z)} such that C{x} is smaller than C{y}\n\n            2. C{(z, *) + (x, y)} such that C{z} is larger than C{y}\n\n        (Other cases, such as C{y < x < z}, can be split into these\n        two cases; for example C{(y - 1, y)} + C{(x, x) + (z, z + 1)})\n\n        In case 1, C{* > y} and C{* > z}, so C{(x, *) + (y, z) = (x,\n        *)}\n\n        In case 2, C{z > x and z > y}, so the intervals do not merge,\n        and the ranges are sorted as C{[(x, y), (z, *)]}.  C{*} is\n        represented as C{(*, *)}, so this is the same as 2.  but with\n        a C{z} that is greater than everything.\n\n        The result is that there is a maximum of two L{None}s, and one\n        of them has to be the high element in the last tuple in\n        C{self.ranges}.  That means checking if C{self.ranges[-1][-1]}\n        is L{None} suffices to check if I{any} element is L{None}.\n\n        @return: L{True} if L{None} is in some range in ranges and\n            L{False} if otherwise.\n        '
        return self.ranges[-1][-1] is None

    def __contains__(self, value):
        if False:
            while True:
                i = 10
        '\n        May raise TypeError if we encounter an open-ended range\n\n        @param value: Is this in our ranges?\n        @type value: L{int}\n        '
        if self._noneInRanges():
            raise TypeError("Can't determine membership; last value not set")
        for (low, high) in self.ranges:
            if low <= value <= high:
                return True
        return False

    def _iterator(self):
        if False:
            for i in range(10):
                print('nop')
        for (l, h) in self.ranges:
            l = self.getnext(l - 1)
            while l <= h:
                yield l
                l = self.getnext(l)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        if self._noneInRanges():
            raise TypeError("Can't iterate; last value not set")
        return self._iterator()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        res = 0
        for (l, h) in self.ranges:
            if l is None:
                res += 1
            elif h is None:
                raise TypeError("Can't size object; last value not set")
            else:
                res += h - l + 1
        return res

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        p = []
        for (low, high) in self.ranges:
            if low == high:
                if low is None:
                    p.append('*')
                else:
                    p.append(str(low))
            elif high is None:
                p.append('%d:*' % (low,))
            else:
                p.append('%d:%d' % (low, high))
        return ','.join(p)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'<MessageSet {str(self)}>'

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        if isinstance(other, MessageSet):
            return cast(bool, self.ranges == other.ranges)
        return NotImplemented

class LiteralString:

    def __init__(self, size, defered):
        if False:
            for i in range(10):
                print('nop')
        self.size = size
        self.data = []
        self.defer = defered

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.size -= len(data)
        passon = None
        if self.size > 0:
            self.data.append(data)
        else:
            if self.size:
                (data, passon) = (data[:self.size], data[self.size:])
            else:
                passon = b''
            if data:
                self.data.append(data)
        return passon

    def callback(self, line):
        if False:
            print('Hello World!')
        '\n        Call deferred with data and rest of line\n        '
        self.defer.callback((b''.join(self.data), line))

class LiteralFile:
    _memoryFileLimit = 1024 * 1024 * 10

    def __init__(self, size, defered):
        if False:
            i = 10
            return i + 15
        self.size = size
        self.defer = defered
        if size > self._memoryFileLimit:
            self.data = tempfile.TemporaryFile()
        else:
            self.data = BytesIO()

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.size -= len(data)
        passon = None
        if self.size > 0:
            self.data.write(data)
        else:
            if self.size:
                (data, passon) = (data[:self.size], data[self.size:])
            else:
                passon = b''
            if data:
                self.data.write(data)
        return passon

    def callback(self, line):
        if False:
            print('Hello World!')
        '\n        Call deferred with data and rest of line\n        '
        self.data.seek(0, 0)
        self.defer.callback((self.data, line))

class WriteBuffer:
    """
    Buffer up a bunch of writes before sending them all to a transport at once.
    """

    def __init__(self, transport, size=8192):
        if False:
            print('Hello World!')
        self.bufferSize = size
        self.transport = transport
        self._length = 0
        self._writes = []

    def write(self, s):
        if False:
            print('Hello World!')
        self._length += len(s)
        self._writes.append(s)
        if self._length > self.bufferSize:
            self.flush()

    def flush(self):
        if False:
            return 10
        if self._writes:
            self.transport.writeSequence(self._writes)
            self._writes = []
            self._length = 0

class Command:
    _1_RESPONSES = (b'CAPABILITY', b'FLAGS', b'LIST', b'LSUB', b'STATUS', b'SEARCH', b'NAMESPACE')
    _2_RESPONSES = (b'EXISTS', b'EXPUNGE', b'FETCH', b'RECENT')
    _OK_RESPONSES = (b'UIDVALIDITY', b'UNSEEN', b'READ-WRITE', b'READ-ONLY', b'UIDNEXT', b'PERMANENTFLAGS')
    defer = None

    def __init__(self, command, args=None, wantResponse=(), continuation=None, *contArgs, **contKw):
        if False:
            print('Hello World!')
        self.command = command
        self.args = args
        self.wantResponse = wantResponse
        self.continuation = lambda x: continuation(x, *contArgs, **contKw)
        self.lines = []

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '<imap4.Command {!r} {!r} {!r} {!r} {!r}>'.format(self.command, self.args, self.wantResponse, self.continuation, self.lines)

    def format(self, tag):
        if False:
            for i in range(10):
                print('nop')
        if self.args is None:
            return b' '.join((tag, self.command))
        return b' '.join((tag, self.command, self.args))

    def finish(self, lastLine, unusedCallback):
        if False:
            while True:
                i = 10
        send = []
        unuse = []
        for L in self.lines:
            names = parseNestedParens(L)
            N = len(names)
            if N >= 1 and names[0] in self._1_RESPONSES or (N >= 2 and names[1] in self._2_RESPONSES) or (N >= 2 and names[0] == b'OK' and isinstance(names[1], list) and (names[1][0] in self._OK_RESPONSES)):
                send.append(names)
            else:
                unuse.append(names)
        (d, self.defer) = (self.defer, None)
        d.callback((send, lastLine))
        if unuse:
            unusedCallback(unuse)
_SP = b' '
_CTL = bytes(chain(range(33), range(128, 256)))
_nonAtomChars = b']\\\\(){%*"' + _SP + _CTL
_nativeNonAtomChars = _nonAtomChars.decode('charmap')
_nonAtomRE = re.compile('[' + _nativeNonAtomChars + ']')
_atomChars = bytes((ch for ch in range(256) if ch not in _nonAtomChars))

@implementer(IMailboxListener)
class IMAP4Server(basic.LineReceiver, policies.TimeoutMixin):
    """
    Protocol implementation for an IMAP4rev1 server.

    The server can be in any of four states:
        - Non-authenticated
        - Authenticated
        - Selected
        - Logout
    """
    IDENT = b'Twisted IMAP4rev1 Ready'
    timeOut = 60
    POSTAUTH_TIMEOUT = 60 * 30
    startedTLS = False
    canStartTLS = False
    tags = None
    portal = None
    account = None
    _onLogout = None
    mbox = None
    _pendingLiteral = None
    _literalStringLimit = 4096
    challengers = None
    _requiresLastMessageInfo = {b'OR', b'NOT', b'UID'}
    state = 'unauth'
    parseState = 'command'

    def __init__(self, chal=None, contextFactory=None, scheduler=None):
        if False:
            i = 10
            return i + 15
        if chal is None:
            chal = {}
        self.challengers = chal
        self.ctx = contextFactory
        if scheduler is None:
            scheduler = iterateInReactor
        self._scheduler = scheduler
        self._queuedAsync = []

    def capabilities(self):
        if False:
            i = 10
            return i + 15
        cap = {b'AUTH': list(self.challengers.keys())}
        if self.ctx and self.canStartTLS:
            if not self.startedTLS and interfaces.ISSLTransport(self.transport, None) is None:
                cap[b'LOGINDISABLED'] = None
                cap[b'STARTTLS'] = None
        cap[b'NAMESPACE'] = None
        cap[b'IDLE'] = None
        return cap

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.tags = {}
        self.canStartTLS = interfaces.ITLSTransport(self.transport, None) is not None
        self.setTimeout(self.timeOut)
        self.sendServerGreeting()

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        self.setTimeout(None)
        if self._onLogout:
            self._onLogout()
            self._onLogout = None

    def timeoutConnection(self):
        if False:
            print('Hello World!')
        self.sendLine(b'* BYE Autologout; connection idle too long')
        self.transport.loseConnection()
        if self.mbox:
            self.mbox.removeListener(self)
            cmbx = ICloseableMailbox(self.mbox, None)
            if cmbx is not None:
                maybeDeferred(cmbx.close).addErrback(log.err)
            self.mbox = None
        self.state = 'timeout'

    def rawDataReceived(self, data):
        if False:
            return 10
        self.resetTimeout()
        passon = self._pendingLiteral.write(data)
        if passon is not None:
            self.setLineMode(passon)
    blocked = None

    def _unblock(self):
        if False:
            for i in range(10):
                print('nop')
        commands = self.blocked
        self.blocked = None
        while commands and self.blocked is None:
            self.lineReceived(commands.pop(0))
        if self.blocked is not None:
            self.blocked.extend(commands)

    def lineReceived(self, line):
        if False:
            print('Hello World!')
        if self.blocked is not None:
            self.blocked.append(line)
            return
        self.resetTimeout()
        f = getattr(self, 'parse_' + self.parseState)
        try:
            f(line)
        except Exception as e:
            self.sendUntaggedResponse(b'BAD Server error: ' + networkString(str(e)))
            log.err()

    def parse_command(self, line):
        if False:
            return 10
        args = line.split(None, 2)
        rest = None
        if len(args) == 3:
            (tag, cmd, rest) = args
        elif len(args) == 2:
            (tag, cmd) = args
        elif len(args) == 1:
            tag = args[0]
            self.sendBadResponse(tag, b'Missing command')
            return None
        else:
            self.sendBadResponse(None, b'Null command')
            return None
        cmd = cmd.upper()
        try:
            return self.dispatchCommand(tag, cmd, rest)
        except IllegalClientResponse as e:
            self.sendBadResponse(tag, b'Illegal syntax: ' + networkString(str(e)))
        except IllegalOperation as e:
            self.sendNegativeResponse(tag, b'Illegal operation: ' + networkString(str(e)))
        except IllegalMailboxEncoding as e:
            self.sendNegativeResponse(tag, b'Illegal mailbox name: ' + networkString(str(e)))

    def parse_pending(self, line):
        if False:
            for i in range(10):
                print('nop')
        d = self._pendingLiteral
        self._pendingLiteral = None
        self.parseState = 'command'
        d.callback(line)

    def dispatchCommand(self, tag, cmd, rest, uid=None):
        if False:
            while True:
                i = 10
        f = self.lookupCommand(cmd)
        if f:
            fn = f[0]
            parseargs = f[1:]
            self.__doCommand(tag, fn, [self, tag], parseargs, rest, uid)
        else:
            self.sendBadResponse(tag, b'Unsupported command')

    def lookupCommand(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self, '_'.join((self.state, nativeString(cmd.upper()))), None)

    def __doCommand(self, tag, handler, args, parseargs, line, uid):
        if False:
            for i in range(10):
                print('nop')
        for (i, arg) in enumerate(parseargs):
            if callable(arg):
                parseargs = parseargs[i + 1:]
                maybeDeferred(arg, self, line).addCallback(self.__cbDispatch, tag, handler, args, parseargs, uid).addErrback(self.__ebDispatch, tag)
                return
            else:
                args.append(arg)
        if line:
            raise IllegalClientResponse('Too many arguments for command: ' + repr(line))
        if uid is not None:
            handler(*args, uid=uid)
        else:
            handler(*args)

    def __cbDispatch(self, result, tag, fn, args, parseargs, uid):
        if False:
            for i in range(10):
                print('nop')
        (arg, rest) = result
        args.append(arg)
        self.__doCommand(tag, fn, args, parseargs, rest, uid)

    def __ebDispatch(self, failure, tag):
        if False:
            for i in range(10):
                print('nop')
        if failure.check(IllegalClientResponse):
            self.sendBadResponse(tag, b'Illegal syntax: ' + networkString(str(failure.value)))
        elif failure.check(IllegalOperation):
            self.sendNegativeResponse(tag, b'Illegal operation: ' + networkString(str(failure.value)))
        elif failure.check(IllegalMailboxEncoding):
            self.sendNegativeResponse(tag, b'Illegal mailbox name: ' + networkString(str(failure.value)))
        else:
            self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))
            log.err(failure)

    def _stringLiteral(self, size):
        if False:
            print('Hello World!')
        if size > self._literalStringLimit:
            raise IllegalClientResponse('Literal too long! I accept at most %d octets' % (self._literalStringLimit,))
        d = defer.Deferred()
        self.parseState = 'pending'
        self._pendingLiteral = LiteralString(size, d)
        self.sendContinuationRequest(networkString('Ready for %d octets of text' % size))
        self.setRawMode()
        return d

    def _fileLiteral(self, size):
        if False:
            i = 10
            return i + 15
        d = defer.Deferred()
        self.parseState = 'pending'
        self._pendingLiteral = LiteralFile(size, d)
        self.sendContinuationRequest(networkString('Ready for %d octets of data' % size))
        self.setRawMode()
        return d

    def arg_finalastring(self, line):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parse an astring from line that represents a command's final\n        argument.  This special case exists to enable parsing empty\n        string literals.\n\n        @param line: A line that contains a string literal.\n        @type line: L{bytes}\n\n        @return: A 2-tuple containing the parsed argument and any\n            trailing data, or a L{Deferred} that fires with that\n            2-tuple\n        @rtype: L{tuple} of (L{bytes}, L{bytes}) or a L{Deferred}\n\n        @see: https://twistedmatrix.com/trac/ticket/9207\n        "
        return self.arg_astring(line, final=True)

    def arg_astring(self, line, final=False):
        if False:
            i = 10
            return i + 15
        '\n        Parse an astring from the line, return (arg, rest), possibly\n        via a deferred (to handle literals)\n\n        @param line: A line that contains a string literal.\n        @type line: L{bytes}\n\n        @param final: Is this the final argument?\n        @type final L{bool}\n\n        @return: A 2-tuple containing the parsed argument and any\n            trailing data, or a L{Deferred} that fires with that\n            2-tuple\n        @rtype: L{tuple} of (L{bytes}, L{bytes}) or a L{Deferred}\n\n        '
        line = line.strip()
        if not line:
            raise IllegalClientResponse('Missing argument')
        d = None
        (arg, rest) = (None, None)
        if line[0:1] == b'"':
            try:
                (spam, arg, rest) = line.split(b'"', 2)
                rest = rest[1:]
            except ValueError:
                raise IllegalClientResponse('Unmatched quotes')
        elif line[0:1] == b'{':
            if line[-1:] != b'}':
                raise IllegalClientResponse('Malformed literal')
            try:
                size = int(line[1:-1])
            except ValueError:
                raise IllegalClientResponse('Bad literal size: ' + repr(line[1:-1]))
            if final and (not size):
                return (b'', b'')
            d = self._stringLiteral(size)
        else:
            arg = line.split(b' ', 1)
            if len(arg) == 1:
                arg.append(b'')
            (arg, rest) = arg
        return d or (arg, rest)
    atomre = re.compile(b'(?P<atom>[' + re.escape(_atomChars) + b']+)( (?P<rest>.*$)|$)')

    def arg_atom(self, line):
        if False:
            return 10
        '\n        Parse an atom from the line\n        '
        if not line:
            raise IllegalClientResponse('Missing argument')
        m = self.atomre.match(line)
        if m:
            return (m.group('atom'), m.group('rest'))
        else:
            raise IllegalClientResponse('Malformed ATOM')

    def arg_plist(self, line):
        if False:
            while True:
                i = 10
        '\n        Parse a (non-nested) parenthesised list from the line\n        '
        if not line:
            raise IllegalClientResponse('Missing argument')
        if line[:1] != b'(':
            raise IllegalClientResponse('Missing parenthesis')
        i = line.find(b')')
        if i == -1:
            raise IllegalClientResponse('Mismatched parenthesis')
        return (parseNestedParens(line[1:i], 0), line[i + 2:])

    def arg_literal(self, line):
        if False:
            print('Hello World!')
        '\n        Parse a literal from the line\n        '
        if not line:
            raise IllegalClientResponse('Missing argument')
        if line[:1] != b'{':
            raise IllegalClientResponse('Missing literal')
        if line[-1:] != b'}':
            raise IllegalClientResponse('Malformed literal')
        try:
            size = int(line[1:-1])
        except ValueError:
            raise IllegalClientResponse(f'Bad literal size: {line[1:-1]!r}')
        return self._fileLiteral(size)

    def arg_searchkeys(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        searchkeys\n        '
        query = parseNestedParens(line)
        return (query, b'')

    def arg_seqset(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        sequence-set\n        '
        rest = b''
        arg = line.split(b' ', 1)
        if len(arg) == 2:
            rest = arg[1]
        arg = arg[0]
        try:
            return (parseIdList(arg), rest)
        except IllegalIdentifierError as e:
            raise IllegalClientResponse('Bad message number ' + str(e))

    def arg_fetchatt(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        fetch-att\n        '
        p = _FetchParser()
        p.parseString(line)
        return (p.result, b'')

    def arg_flaglist(self, line):
        if False:
            return 10
        '\n        Flag part of store-att-flag\n        '
        flags = []
        if line[0:1] == b'(':
            if line[-1:] != b')':
                raise IllegalClientResponse('Mismatched parenthesis')
            line = line[1:-1]
        while line:
            m = self.atomre.search(line)
            if not m:
                raise IllegalClientResponse('Malformed flag')
            if line[0:1] == b'\\' and m.start() == 1:
                flags.append(b'\\' + m.group('atom'))
            elif m.start() == 0:
                flags.append(m.group('atom'))
            else:
                raise IllegalClientResponse('Malformed flag')
            line = m.group('rest')
        return (flags, b'')

    def arg_line(self, line):
        if False:
            return 10
        '\n        Command line of UID command\n        '
        return (line, b'')

    def opt_plist(self, line):
        if False:
            while True:
                i = 10
        '\n        Optional parenthesised list\n        '
        if line.startswith(b'('):
            return self.arg_plist(line)
        else:
            return (None, line)

    def opt_datetime(self, line):
        if False:
            return 10
        '\n        Optional date-time string\n        '
        if line.startswith(b'"'):
            try:
                (spam, date, rest) = line.split(b'"', 2)
            except ValueError:
                raise IllegalClientResponse('Malformed date-time')
            return (date, rest[1:])
        else:
            return (None, line)

    def opt_charset(self, line):
        if False:
            return 10
        '\n        Optional charset of SEARCH command\n        '
        if line[:7].upper() == b'CHARSET':
            arg = line.split(b' ', 2)
            if len(arg) == 1:
                raise IllegalClientResponse('Missing charset identifier')
            if len(arg) == 2:
                arg.append(b'')
            (spam, arg, rest) = arg
            return (arg, rest)
        else:
            return (None, line)

    def sendServerGreeting(self):
        if False:
            return 10
        msg = b'[CAPABILITY ' + b' '.join(self.listCapabilities()) + b'] ' + self.IDENT
        self.sendPositiveResponse(message=msg)

    def sendBadResponse(self, tag=None, message=b''):
        if False:
            return 10
        self._respond(b'BAD', tag, message)

    def sendPositiveResponse(self, tag=None, message=b''):
        if False:
            while True:
                i = 10
        self._respond(b'OK', tag, message)

    def sendNegativeResponse(self, tag=None, message=b''):
        if False:
            print('Hello World!')
        self._respond(b'NO', tag, message)

    def sendUntaggedResponse(self, message, isAsync=None, **kwargs):
        if False:
            return 10
        isAsync = _get_async_param(isAsync, **kwargs)
        if not isAsync or self.blocked is None:
            self._respond(message, None, None)
        else:
            self._queuedAsync.append(message)

    def sendContinuationRequest(self, msg=b'Ready for additional command text'):
        if False:
            while True:
                i = 10
        if msg:
            self.sendLine(b'+ ' + msg)
        else:
            self.sendLine(b'+')

    def _respond(self, state, tag, message):
        if False:
            i = 10
            return i + 15
        if state in (b'OK', b'NO', b'BAD') and self._queuedAsync:
            lines = self._queuedAsync
            self._queuedAsync = []
            for msg in lines:
                self._respond(msg, None, None)
        if not tag:
            tag = b'*'
        if message:
            self.sendLine(b' '.join((tag, state, message)))
        else:
            self.sendLine(b' '.join((tag, state)))

    def listCapabilities(self):
        if False:
            while True:
                i = 10
        caps = [b'IMAP4rev1']
        for (c, v) in self.capabilities().items():
            if v is None:
                caps.append(c)
            elif len(v):
                caps.extend([c + b'=' + cap for cap in v])
        return caps

    def do_CAPABILITY(self, tag):
        if False:
            return 10
        self.sendUntaggedResponse(b'CAPABILITY ' + b' '.join(self.listCapabilities()))
        self.sendPositiveResponse(tag, b'CAPABILITY completed')
    unauth_CAPABILITY = (do_CAPABILITY,)
    auth_CAPABILITY = unauth_CAPABILITY
    select_CAPABILITY = unauth_CAPABILITY
    logout_CAPABILITY = unauth_CAPABILITY

    def do_LOGOUT(self, tag):
        if False:
            return 10
        self.sendUntaggedResponse(b'BYE Nice talking to you')
        self.sendPositiveResponse(tag, b'LOGOUT successful')
        self.transport.loseConnection()
    unauth_LOGOUT = (do_LOGOUT,)
    auth_LOGOUT = unauth_LOGOUT
    select_LOGOUT = unauth_LOGOUT
    logout_LOGOUT = unauth_LOGOUT

    def do_NOOP(self, tag):
        if False:
            print('Hello World!')
        self.sendPositiveResponse(tag, b'NOOP No operation performed')
    unauth_NOOP = (do_NOOP,)
    auth_NOOP = unauth_NOOP
    select_NOOP = unauth_NOOP
    logout_NOOP = unauth_NOOP

    def do_AUTHENTICATE(self, tag, args):
        if False:
            return 10
        args = args.upper().strip()
        if args not in self.challengers:
            self.sendNegativeResponse(tag, b'AUTHENTICATE method unsupported')
        else:
            self.authenticate(self.challengers[args](), tag)
    unauth_AUTHENTICATE = (do_AUTHENTICATE, arg_atom)

    def authenticate(self, chal, tag):
        if False:
            i = 10
            return i + 15
        if self.portal is None:
            self.sendNegativeResponse(tag, b'Temporary authentication failure')
            return
        self._setupChallenge(chal, tag)

    def _setupChallenge(self, chal, tag):
        if False:
            for i in range(10):
                print('nop')
        try:
            challenge = chal.getChallenge()
        except Exception as e:
            self.sendBadResponse(tag, b'Server error: ' + networkString(str(e)))
        else:
            coded = encodebytes(challenge)[:-1]
            self.parseState = 'pending'
            self._pendingLiteral = defer.Deferred()
            self.sendContinuationRequest(coded)
            self._pendingLiteral.addCallback(self.__cbAuthChunk, chal, tag)
            self._pendingLiteral.addErrback(self.__ebAuthChunk, tag)

    def __cbAuthChunk(self, result, chal, tag):
        if False:
            while True:
                i = 10
        try:
            uncoded = decodebytes(result)
        except binascii.Error:
            raise IllegalClientResponse('Malformed Response - not base64')
        chal.setResponse(uncoded)
        if chal.moreChallenges():
            self._setupChallenge(chal, tag)
        else:
            self.portal.login(chal, None, IAccount).addCallbacks(self.__cbAuthResp, self.__ebAuthResp, (tag,), None, (tag,), None)

    def __cbAuthResp(self, result, tag):
        if False:
            return 10
        (iface, avatar, logout) = result
        assert iface is IAccount, 'IAccount is the only supported interface'
        self.account = avatar
        self.state = 'auth'
        self._onLogout = logout
        self.sendPositiveResponse(tag, b'Authentication successful')
        self.setTimeout(self.POSTAUTH_TIMEOUT)

    def __ebAuthResp(self, failure, tag):
        if False:
            print('Hello World!')
        if failure.check(UnauthorizedLogin):
            self.sendNegativeResponse(tag, b'Authentication failed: unauthorized')
        elif failure.check(UnhandledCredentials):
            self.sendNegativeResponse(tag, b'Authentication failed: server misconfigured')
        else:
            self.sendBadResponse(tag, b'Server error: login failed unexpectedly')
            log.err(failure)

    def __ebAuthChunk(self, failure, tag):
        if False:
            print('Hello World!')
        self.sendNegativeResponse(tag, b'Authentication failed: ' + networkString(str(failure.value)))

    def do_STARTTLS(self, tag):
        if False:
            for i in range(10):
                print('nop')
        if self.startedTLS:
            self.sendNegativeResponse(tag, b'TLS already negotiated')
        elif self.ctx and self.canStartTLS:
            self.sendPositiveResponse(tag, b'Begin TLS negotiation now')
            self.transport.startTLS(self.ctx)
            self.startedTLS = True
            self.challengers = self.challengers.copy()
            if b'LOGIN' not in self.challengers:
                self.challengers[b'LOGIN'] = LOGINCredentials
            if b'PLAIN' not in self.challengers:
                self.challengers[b'PLAIN'] = PLAINCredentials
        else:
            self.sendNegativeResponse(tag, b'TLS not available')
    unauth_STARTTLS = (do_STARTTLS,)

    def do_LOGIN(self, tag, user, passwd):
        if False:
            return 10
        if b'LOGINDISABLED' in self.capabilities():
            self.sendBadResponse(tag, b'LOGIN is disabled before STARTTLS')
            return
        maybeDeferred(self.authenticateLogin, user, passwd).addCallback(self.__cbLogin, tag).addErrback(self.__ebLogin, tag)
    unauth_LOGIN = (do_LOGIN, arg_astring, arg_finalastring)

    def authenticateLogin(self, user, passwd):
        if False:
            return 10
        '\n        Lookup the account associated with the given parameters\n\n        Override this method to define the desired authentication behavior.\n\n        The default behavior is to defer authentication to C{self.portal}\n        if it is not None, or to deny the login otherwise.\n\n        @type user: L{str}\n        @param user: The username to lookup\n\n        @type passwd: L{str}\n        @param passwd: The password to login with\n        '
        if self.portal:
            return self.portal.login(credentials.UsernamePassword(user, passwd), None, IAccount)
        raise UnauthorizedLogin()

    def __cbLogin(self, result, tag):
        if False:
            return 10
        (iface, avatar, logout) = result
        if iface is not IAccount:
            self.sendBadResponse(tag, b'Server error: login returned unexpected value')
            log.err(f'__cbLogin called with {iface!r}, IAccount expected')
        else:
            self.account = avatar
            self._onLogout = logout
            self.sendPositiveResponse(tag, b'LOGIN succeeded')
            self.state = 'auth'
            self.setTimeout(self.POSTAUTH_TIMEOUT)

    def __ebLogin(self, failure, tag):
        if False:
            return 10
        if failure.check(UnauthorizedLogin):
            self.sendNegativeResponse(tag, b'LOGIN failed')
        else:
            self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))
            log.err(failure)

    def do_NAMESPACE(self, tag):
        if False:
            i = 10
            return i + 15
        personal = public = shared = None
        np = INamespacePresenter(self.account, None)
        if np is not None:
            personal = np.getPersonalNamespaces()
            public = np.getSharedNamespaces()
            shared = np.getSharedNamespaces()
        self.sendUntaggedResponse(b'NAMESPACE ' + collapseNestedLists([personal, public, shared]))
        self.sendPositiveResponse(tag, b'NAMESPACE command completed')
    auth_NAMESPACE = (do_NAMESPACE,)
    select_NAMESPACE = auth_NAMESPACE

    def _selectWork(self, tag, name, rw, cmdName):
        if False:
            for i in range(10):
                print('nop')
        if self.mbox:
            self.mbox.removeListener(self)
            cmbx = ICloseableMailbox(self.mbox, None)
            if cmbx is not None:
                maybeDeferred(cmbx.close).addErrback(log.err)
            self.mbox = None
            self.state = 'auth'
        name = _parseMbox(name)
        maybeDeferred(self.account.select, _parseMbox(name), rw).addCallback(self._cbSelectWork, cmdName, tag).addErrback(self._ebSelectWork, cmdName, tag)

    def _ebSelectWork(self, failure, cmdName, tag):
        if False:
            i = 10
            return i + 15
        self.sendBadResponse(tag, cmdName + b' failed: Server error')
        log.err(failure)

    def _cbSelectWork(self, mbox, cmdName, tag):
        if False:
            return 10
        if mbox is None:
            self.sendNegativeResponse(tag, b'No such mailbox')
            return
        if '\\noselect' in [s.lower() for s in mbox.getFlags()]:
            self.sendNegativeResponse(tag, 'Mailbox cannot be selected')
            return
        flags = [networkString(flag) for flag in mbox.getFlags()]
        self.sendUntaggedResponse(b'%d EXISTS' % (mbox.getMessageCount(),))
        self.sendUntaggedResponse(b'%d RECENT' % (mbox.getRecentCount(),))
        self.sendUntaggedResponse(b'FLAGS (' + b' '.join(flags) + b')')
        self.sendPositiveResponse(None, b'[UIDVALIDITY %d]' % (mbox.getUIDValidity(),))
        s = mbox.isWriteable() and b'READ-WRITE' or b'READ-ONLY'
        mbox.addListener(self)
        self.sendPositiveResponse(tag, b'[' + s + b'] ' + cmdName + b' successful')
        self.state = 'select'
        self.mbox = mbox
    auth_SELECT = (_selectWork, arg_astring, 1, b'SELECT')
    select_SELECT = auth_SELECT
    auth_EXAMINE = (_selectWork, arg_astring, 0, b'EXAMINE')
    select_EXAMINE = auth_EXAMINE

    def do_IDLE(self, tag):
        if False:
            i = 10
            return i + 15
        self.sendContinuationRequest(None)
        self.parseTag = tag
        self.lastState = self.parseState
        self.parseState = 'idle'

    def parse_idle(self, *args):
        if False:
            i = 10
            return i + 15
        self.parseState = self.lastState
        del self.lastState
        self.sendPositiveResponse(self.parseTag, b'IDLE terminated')
        del self.parseTag
    select_IDLE = (do_IDLE,)
    auth_IDLE = select_IDLE

    def do_CREATE(self, tag, name):
        if False:
            return 10
        name = _parseMbox(name)
        try:
            result = self.account.create(name)
        except MailboxException as c:
            self.sendNegativeResponse(tag, networkString(str(c)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while creating mailbox')
            log.err()
        else:
            if result:
                self.sendPositiveResponse(tag, b'Mailbox created')
            else:
                self.sendNegativeResponse(tag, b'Mailbox not created')
    auth_CREATE = (do_CREATE, arg_finalastring)
    select_CREATE = auth_CREATE

    def do_DELETE(self, tag, name):
        if False:
            return 10
        name = _parseMbox(name)
        if name.lower() == 'inbox':
            self.sendNegativeResponse(tag, b'You cannot delete the inbox')
            return
        try:
            self.account.delete(name)
        except MailboxException as m:
            self.sendNegativeResponse(tag, str(m).encode('imap4-utf-7'))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while deleting mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Mailbox deleted')
    auth_DELETE = (do_DELETE, arg_finalastring)
    select_DELETE = auth_DELETE

    def do_RENAME(self, tag, oldname, newname):
        if False:
            return 10
        (oldname, newname) = (_parseMbox(n) for n in (oldname, newname))
        if oldname.lower() == 'inbox' or newname.lower() == 'inbox':
            self.sendNegativeResponse(tag, b'You cannot rename the inbox, or rename another mailbox to inbox.')
            return
        try:
            self.account.rename(oldname, newname)
        except TypeError:
            self.sendBadResponse(tag, b'Invalid command syntax')
        except MailboxException as m:
            self.sendNegativeResponse(tag, networkString(str(m)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while renaming mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Mailbox renamed')
    auth_RENAME = (do_RENAME, arg_astring, arg_finalastring)
    select_RENAME = auth_RENAME

    def do_SUBSCRIBE(self, tag, name):
        if False:
            i = 10
            return i + 15
        name = _parseMbox(name)
        try:
            self.account.subscribe(name)
        except MailboxException as m:
            self.sendNegativeResponse(tag, networkString(str(m)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while subscribing to mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Subscribed')
    auth_SUBSCRIBE = (do_SUBSCRIBE, arg_finalastring)
    select_SUBSCRIBE = auth_SUBSCRIBE

    def do_UNSUBSCRIBE(self, tag, name):
        if False:
            i = 10
            return i + 15
        name = _parseMbox(name)
        try:
            self.account.unsubscribe(name)
        except MailboxException as m:
            self.sendNegativeResponse(tag, networkString(str(m)))
        except BaseException:
            self.sendBadResponse(tag, b'Server error encountered while unsubscribing from mailbox')
            log.err()
        else:
            self.sendPositiveResponse(tag, b'Unsubscribed')
    auth_UNSUBSCRIBE = (do_UNSUBSCRIBE, arg_finalastring)
    select_UNSUBSCRIBE = auth_UNSUBSCRIBE

    def _listWork(self, tag, ref, mbox, sub, cmdName):
        if False:
            return 10
        mbox = _parseMbox(mbox)
        ref = _parseMbox(ref)
        maybeDeferred(self.account.listMailboxes, ref, mbox).addCallback(self._cbListWork, tag, sub, cmdName).addErrback(self._ebListWork, tag)

    def _cbListWork(self, mailboxes, tag, sub, cmdName):
        if False:
            while True:
                i = 10
        for (name, box) in mailboxes:
            if not sub or self.account.isSubscribed(name):
                flags = [networkString(flag) for flag in box.getFlags()]
                delim = box.getHierarchicalDelimiter().encode('imap4-utf-7')
                resp = (DontQuoteMe(cmdName), map(DontQuoteMe, flags), delim, name.encode('imap4-utf-7'))
                self.sendUntaggedResponse(collapseNestedLists(resp))
        self.sendPositiveResponse(tag, cmdName + b' completed')

    def _ebListWork(self, failure, tag):
        if False:
            for i in range(10):
                print('nop')
        self.sendBadResponse(tag, b'Server error encountered while listing mailboxes.')
        log.err(failure)
    auth_LIST = (_listWork, arg_astring, arg_astring, 0, b'LIST')
    select_LIST = auth_LIST
    auth_LSUB = (_listWork, arg_astring, arg_astring, 1, b'LSUB')
    select_LSUB = auth_LSUB

    def do_STATUS(self, tag, mailbox, names):
        if False:
            for i in range(10):
                print('nop')
        nativeNames = []
        for name in names:
            nativeNames.append(nativeString(name))
        mailbox = _parseMbox(mailbox)
        maybeDeferred(self.account.select, mailbox, 0).addCallback(self._cbStatusGotMailbox, tag, mailbox, nativeNames).addErrback(self._ebStatusGotMailbox, tag)

    def _cbStatusGotMailbox(self, mbox, tag, mailbox, names):
        if False:
            while True:
                i = 10
        if mbox:
            maybeDeferred(mbox.requestStatus, names).addCallbacks(self.__cbStatus, self.__ebStatus, (tag, mailbox), None, (tag, mailbox), None)
        else:
            self.sendNegativeResponse(tag, b'Could not open mailbox')

    def _ebStatusGotMailbox(self, failure, tag):
        if False:
            i = 10
            return i + 15
        self.sendBadResponse(tag, b'Server error encountered while opening mailbox.')
        log.err(failure)
    auth_STATUS = (do_STATUS, arg_astring, arg_plist)
    select_STATUS = auth_STATUS

    def __cbStatus(self, status, tag, box):
        if False:
            return 10
        line = networkString(' '.join(['%s %s' % x for x in status.items()]))
        self.sendUntaggedResponse(b'STATUS ' + box.encode('imap4-utf-7') + b' (' + line + b')')
        self.sendPositiveResponse(tag, b'STATUS complete')

    def __ebStatus(self, failure, tag, box):
        if False:
            return 10
        self.sendBadResponse(tag, b'STATUS ' + box + b' failed: ' + networkString(str(failure.value)))

    def do_APPEND(self, tag, mailbox, flags, date, message):
        if False:
            for i in range(10):
                print('nop')
        mailbox = _parseMbox(mailbox)
        maybeDeferred(self.account.select, mailbox).addCallback(self._cbAppendGotMailbox, tag, flags, date, message).addErrback(self._ebAppendGotMailbox, tag)

    def _cbAppendGotMailbox(self, mbox, tag, flags, date, message):
        if False:
            print('Hello World!')
        if not mbox:
            self.sendNegativeResponse(tag, '[TRYCREATE] No such mailbox')
            return
        decodedFlags = [nativeString(flag) for flag in flags]
        d = mbox.addMessage(message, decodedFlags, date)
        d.addCallback(self.__cbAppend, tag, mbox)
        d.addErrback(self.__ebAppend, tag)

    def _ebAppendGotMailbox(self, failure, tag):
        if False:
            return 10
        self.sendBadResponse(tag, b'Server error encountered while opening mailbox.')
        log.err(failure)
    auth_APPEND = (do_APPEND, arg_astring, opt_plist, opt_datetime, arg_literal)
    select_APPEND = auth_APPEND

    def __cbAppend(self, result, tag, mbox):
        if False:
            return 10
        self.sendUntaggedResponse(b'%d EXISTS' % (mbox.getMessageCount(),))
        self.sendPositiveResponse(tag, b'APPEND complete')

    def __ebAppend(self, failure, tag):
        if False:
            for i in range(10):
                print('nop')
        self.sendBadResponse(tag, b'APPEND failed: ' + networkString(str(failure.value)))

    def do_CHECK(self, tag):
        if False:
            for i in range(10):
                print('nop')
        d = self.checkpoint()
        if d is None:
            self.__cbCheck(None, tag)
        else:
            d.addCallbacks(self.__cbCheck, self.__ebCheck, callbackArgs=(tag,), errbackArgs=(tag,))
    select_CHECK = (do_CHECK,)

    def __cbCheck(self, result, tag):
        if False:
            i = 10
            return i + 15
        self.sendPositiveResponse(tag, b'CHECK completed')

    def __ebCheck(self, failure, tag):
        if False:
            for i in range(10):
                print('nop')
        self.sendBadResponse(tag, b'CHECK failed: ' + networkString(str(failure.value)))

    def checkpoint(self):
        if False:
            i = 10
            return i + 15
        "\n        Called when the client issues a CHECK command.\n\n        This should perform any checkpoint operations required by the server.\n        It may be a long running operation, but may not block.  If it returns\n        a deferred, the client will only be informed of success (or failure)\n        when the deferred's callback (or errback) is invoked.\n        "
        return None

    def do_CLOSE(self, tag):
        if False:
            for i in range(10):
                print('nop')
        d = None
        if self.mbox.isWriteable():
            d = maybeDeferred(self.mbox.expunge)
        cmbx = ICloseableMailbox(self.mbox, None)
        if cmbx is not None:
            if d is not None:
                d.addCallback(lambda result: cmbx.close())
            else:
                d = maybeDeferred(cmbx.close)
        if d is not None:
            d.addCallbacks(self.__cbClose, self.__ebClose, (tag,), None, (tag,), None)
        else:
            self.__cbClose(None, tag)
    select_CLOSE = (do_CLOSE,)

    def __cbClose(self, result, tag):
        if False:
            return 10
        self.sendPositiveResponse(tag, b'CLOSE completed')
        self.mbox.removeListener(self)
        self.mbox = None
        self.state = 'auth'

    def __ebClose(self, failure, tag):
        if False:
            print('Hello World!')
        self.sendBadResponse(tag, b'CLOSE failed: ' + networkString(str(failure.value)))

    def do_EXPUNGE(self, tag):
        if False:
            return 10
        if self.mbox.isWriteable():
            maybeDeferred(self.mbox.expunge).addCallbacks(self.__cbExpunge, self.__ebExpunge, (tag,), None, (tag,), None)
        else:
            self.sendNegativeResponse(tag, b'EXPUNGE ignored on read-only mailbox')
    select_EXPUNGE = (do_EXPUNGE,)

    def __cbExpunge(self, result, tag):
        if False:
            return 10
        for e in result:
            self.sendUntaggedResponse(b'%d EXPUNGE' % (e,))
        self.sendPositiveResponse(tag, b'EXPUNGE completed')

    def __ebExpunge(self, failure, tag):
        if False:
            return 10
        self.sendBadResponse(tag, b'EXPUNGE failed: ' + networkString(str(failure.value)))
        log.err(failure)

    def do_SEARCH(self, tag, charset, query, uid=0):
        if False:
            return 10
        sm = ISearchableMailbox(self.mbox, None)
        if sm is not None:
            maybeDeferred(sm.search, query, uid=uid).addCallback(self.__cbSearch, tag, self.mbox, uid).addErrback(self.__ebSearch, tag)
        else:
            s = parseIdList(b'1:*')
            maybeDeferred(self.mbox.fetch, s, uid=uid).addCallback(self.__cbManualSearch, tag, self.mbox, query, uid).addErrback(self.__ebSearch, tag)
    select_SEARCH = (do_SEARCH, opt_charset, arg_searchkeys)

    def __cbSearch(self, result, tag, mbox, uid):
        if False:
            return 10
        if uid:
            result = map(mbox.getUID, result)
        ids = networkString(' '.join([str(i) for i in result]))
        self.sendUntaggedResponse(b'SEARCH ' + ids)
        self.sendPositiveResponse(tag, b'SEARCH completed')

    def __cbManualSearch(self, result, tag, mbox, query, uid, searchResults=None):
        if False:
            print('Hello World!')
        '\n        Apply the search filter to a set of messages. Send the response to the\n        client.\n\n        @type result: L{list} of L{tuple} of (L{int}, provider of\n            L{imap4.IMessage})\n        @param result: A list two tuples of messages with their sequence ids,\n            sorted by the ids in descending order.\n\n        @type tag: L{str}\n        @param tag: A command tag.\n\n        @type mbox: Provider of L{imap4.IMailbox}\n        @param mbox: The searched mailbox.\n\n        @type query: L{list}\n        @param query: A list representing the parsed form of the search query.\n\n        @param uid: A flag indicating whether the search is over message\n            sequence numbers or UIDs.\n\n        @type searchResults: L{list}\n        @param searchResults: The search results so far or L{None} if no\n            results yet.\n        '
        if searchResults is None:
            searchResults = []
        i = 0
        lastSequenceId = result and result[-1][0]
        lastMessageId = result and result[-1][1].getUID()
        for (i, (msgId, msg)) in list(zip(range(5), result)):
            if self._searchFilter(copy.deepcopy(query), msgId, msg, lastSequenceId, lastMessageId):
                searchResults.append(b'%d' % (msg.getUID() if uid else msgId,))
        if i == 4:
            from twisted.internet import reactor
            reactor.callLater(0, self.__cbManualSearch, list(result[5:]), tag, mbox, query, uid, searchResults)
        else:
            if searchResults:
                self.sendUntaggedResponse(b'SEARCH ' + b' '.join(searchResults))
            self.sendPositiveResponse(tag, b'SEARCH completed')

    def _searchFilter(self, query, id, msg, lastSequenceId, lastMessageId):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pop search terms from the beginning of C{query} until there are none\n        left and apply them to the given message.\n\n        @param query: A list representing the parsed form of the search query.\n\n        @param id: The sequence number of the message being checked.\n\n        @param msg: The message being checked.\n\n        @type lastSequenceId: L{int}\n        @param lastSequenceId: The highest sequence number of any message in\n            the mailbox being searched.\n\n        @type lastMessageId: L{int}\n        @param lastMessageId: The highest UID of any message in the mailbox\n            being searched.\n\n        @return: Boolean indicating whether all of the query terms match the\n            message.\n        '
        while query:
            if not self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId):
                return False
        return True

    def _singleSearchStep(self, query, msgId, msg, lastSequenceId, lastMessageId):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pop one search term from the beginning of C{query} (possibly more than\n        one element) and return whether it matches the given message.\n\n        @param query: A list representing the parsed form of the search query.\n\n        @param msgId: The sequence number of the message being checked.\n\n        @param msg: The message being checked.\n\n        @param lastSequenceId: The highest sequence number of any message in\n            the mailbox being searched.\n\n        @param lastMessageId: The highest UID of any message in the mailbox\n            being searched.\n\n        @return: Boolean indicating whether the query term matched the message.\n        '
        q = query.pop(0)
        if isinstance(q, list):
            if not self._searchFilter(q, msgId, msg, lastSequenceId, lastMessageId):
                return False
        else:
            c = q.upper()
            if not c[:1].isalpha():
                messageSet = parseIdList(c, lastSequenceId)
                return msgId in messageSet
            else:
                f = getattr(self, 'search_' + nativeString(c), None)
                if f is None:
                    raise IllegalQueryError('Invalid search command %s' % nativeString(c))
                if c in self._requiresLastMessageInfo:
                    result = f(query, msgId, msg, (lastSequenceId, lastMessageId))
                else:
                    result = f(query, msgId, msg)
                if not result:
                    return False
        return True

    def search_ALL(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        '\n        Returns C{True} if the message matches the ALL search key (always).\n\n        @type query: A L{list} of L{str}\n        @param query: A list representing the parsed query string.\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        '
        return True

    def search_ANSWERED(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        '\n        Returns C{True} if the message has been answered.\n\n        @type query: A L{list} of L{str}\n        @param query: A list representing the parsed query string.\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        '
        return '\\Answered' in msg.getFlags()

    def search_BCC(self, query, id, msg):
        if False:
            print('Hello World!')
        '\n        Returns C{True} if the message has a BCC address matching the query.\n\n        @type query: A L{list} of L{str}\n        @param query: A list whose first element is a BCC L{str}\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        '
        bcc = msg.getHeaders(False, 'bcc').get('bcc', '')
        return bcc.lower().find(query.pop(0).lower()) != -1

    def search_BEFORE(self, query, id, msg):
        if False:
            while True:
                i = 10
        date = parseTime(query.pop(0))
        return email.utils.parsedate(nativeString(msg.getInternalDate())) < date

    def search_BODY(self, query, id, msg):
        if False:
            while True:
                i = 10
        body = query.pop(0).lower()
        return text.strFile(body, msg.getBodyFile(), False)

    def search_CC(self, query, id, msg):
        if False:
            while True:
                i = 10
        cc = msg.getHeaders(False, 'cc').get('cc', '')
        return cc.lower().find(query.pop(0).lower()) != -1

    def search_DELETED(self, query, id, msg):
        if False:
            while True:
                i = 10
        return '\\Deleted' in msg.getFlags()

    def search_DRAFT(self, query, id, msg):
        if False:
            print('Hello World!')
        return '\\Draft' in msg.getFlags()

    def search_FLAGGED(self, query, id, msg):
        if False:
            for i in range(10):
                print('nop')
        return '\\Flagged' in msg.getFlags()

    def search_FROM(self, query, id, msg):
        if False:
            print('Hello World!')
        fm = msg.getHeaders(False, 'from').get('from', '')
        return fm.lower().find(query.pop(0).lower()) != -1

    def search_HEADER(self, query, id, msg):
        if False:
            while True:
                i = 10
        hdr = query.pop(0).lower()
        hdr = msg.getHeaders(False, hdr).get(hdr, '')
        return hdr.lower().find(query.pop(0).lower()) != -1

    def search_KEYWORD(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        query.pop(0)
        return False

    def search_LARGER(self, query, id, msg):
        if False:
            return 10
        return int(query.pop(0)) < msg.getSize()

    def search_NEW(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        return '\\Recent' in msg.getFlags() and '\\Seen' not in msg.getFlags()

    def search_NOT(self, query, id, msg, lastIDs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns C{True} if the message does not match the query.\n\n        @type query: A L{list} of L{str}\n        @param query: A list representing the parsed form of the search query.\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        @param msg: The message being checked.\n\n        @type lastIDs: L{tuple}\n        @param lastIDs: A tuple of (last sequence id, last message id).\n        The I{last sequence id} is an L{int} containing the highest sequence\n        number of a message in the mailbox.  The I{last message id} is an\n        L{int} containing the highest UID of a message in the mailbox.\n        '
        (lastSequenceId, lastMessageId) = lastIDs
        return not self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)

    def search_OLD(self, query, id, msg):
        if False:
            print('Hello World!')
        return '\\Recent' not in msg.getFlags()

    def search_ON(self, query, id, msg):
        if False:
            for i in range(10):
                print('nop')
        date = parseTime(query.pop(0))
        return email.utils.parsedate(msg.getInternalDate()) == date

    def search_OR(self, query, id, msg, lastIDs):
        if False:
            while True:
                i = 10
        '\n        Returns C{True} if the message matches any of the first two query\n        items.\n\n        @type query: A L{list} of L{str}\n        @param query: A list representing the parsed form of the search query.\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        @param msg: The message being checked.\n\n        @type lastIDs: L{tuple}\n        @param lastIDs: A tuple of (last sequence id, last message id).\n        The I{last sequence id} is an L{int} containing the highest sequence\n        number of a message in the mailbox.  The I{last message id} is an\n        L{int} containing the highest UID of a message in the mailbox.\n        '
        (lastSequenceId, lastMessageId) = lastIDs
        a = self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)
        b = self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId)
        return a or b

    def search_RECENT(self, query, id, msg):
        if False:
            return 10
        return '\\Recent' in msg.getFlags()

    def search_SEEN(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        return '\\Seen' in msg.getFlags()

    def search_SENTBEFORE(self, query, id, msg):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns C{True} if the message date is earlier than the query date.\n\n        @type query: A L{list} of L{str}\n        @param query: A list whose first element starts with a stringified date\n            that is a fragment of an L{imap4.Query()}. The date must be in the\n            format 'DD-Mon-YYYY', for example '03-March-2003' or '03-Mar-2003'.\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        "
        date = msg.getHeaders(False, 'date').get('date', '')
        date = email.utils.parsedate(date)
        return date < parseTime(query.pop(0))

    def search_SENTON(self, query, id, msg):
        if False:
            return 10
        "\n        Returns C{True} if the message date is the same as the query date.\n\n        @type query: A L{list} of L{str}\n        @param query: A list whose first element starts with a stringified date\n            that is a fragment of an L{imap4.Query()}. The date must be in the\n            format 'DD-Mon-YYYY', for example '03-March-2003' or '03-Mar-2003'.\n\n        @type msg: Provider of L{imap4.IMessage}\n        "
        date = msg.getHeaders(False, 'date').get('date', '')
        date = email.utils.parsedate(date)
        return date[:3] == parseTime(query.pop(0))[:3]

    def search_SENTSINCE(self, query, id, msg):
        if False:
            while True:
                i = 10
        "\n        Returns C{True} if the message date is later than the query date.\n\n        @type query: A L{list} of L{str}\n        @param query: A list whose first element starts with a stringified date\n            that is a fragment of an L{imap4.Query()}. The date must be in the\n            format 'DD-Mon-YYYY', for example '03-March-2003' or '03-Mar-2003'.\n\n        @type msg: Provider of L{imap4.IMessage}\n        "
        date = msg.getHeaders(False, 'date').get('date', '')
        date = email.utils.parsedate(date)
        return date > parseTime(query.pop(0))

    def search_SINCE(self, query, id, msg):
        if False:
            for i in range(10):
                print('nop')
        date = parseTime(query.pop(0))
        return email.utils.parsedate(msg.getInternalDate()) > date

    def search_SMALLER(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        return int(query.pop(0)) > msg.getSize()

    def search_SUBJECT(self, query, id, msg):
        if False:
            return 10
        subj = msg.getHeaders(False, 'subject').get('subject', '')
        return subj.lower().find(query.pop(0).lower()) != -1

    def search_TEXT(self, query, id, msg):
        if False:
            print('Hello World!')
        body = query.pop(0).lower()
        return text.strFile(body, msg.getBodyFile(), False)

    def search_TO(self, query, id, msg):
        if False:
            print('Hello World!')
        to = msg.getHeaders(False, 'to').get('to', '')
        return to.lower().find(query.pop(0).lower()) != -1

    def search_UID(self, query, id, msg, lastIDs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns C{True} if the message UID is in the range defined by the\n        search query.\n\n        @type query: A L{list} of L{bytes}\n        @param query: A list representing the parsed form of the search\n            query. Its first element should be a L{str} that can be interpreted\n            as a sequence range, for example '2:4,5:*'.\n\n        @type id: L{int}\n        @param id: The sequence number of the message being checked.\n\n        @type msg: Provider of L{imap4.IMessage}\n        @param msg: The message being checked.\n\n        @type lastIDs: L{tuple}\n        @param lastIDs: A tuple of (last sequence id, last message id).\n        The I{last sequence id} is an L{int} containing the highest sequence\n        number of a message in the mailbox.  The I{last message id} is an\n        L{int} containing the highest UID of a message in the mailbox.\n        "
        (lastSequenceId, lastMessageId) = lastIDs
        c = query.pop(0)
        m = parseIdList(c, lastMessageId)
        return msg.getUID() in m

    def search_UNANSWERED(self, query, id, msg):
        if False:
            i = 10
            return i + 15
        return '\\Answered' not in msg.getFlags()

    def search_UNDELETED(self, query, id, msg):
        if False:
            for i in range(10):
                print('nop')
        return '\\Deleted' not in msg.getFlags()

    def search_UNDRAFT(self, query, id, msg):
        if False:
            for i in range(10):
                print('nop')
        return '\\Draft' not in msg.getFlags()

    def search_UNFLAGGED(self, query, id, msg):
        if False:
            return 10
        return '\\Flagged' not in msg.getFlags()

    def search_UNKEYWORD(self, query, id, msg):
        if False:
            while True:
                i = 10
        query.pop(0)
        return False

    def search_UNSEEN(self, query, id, msg):
        if False:
            return 10
        return '\\Seen' not in msg.getFlags()

    def __ebSearch(self, failure, tag):
        if False:
            return 10
        self.sendBadResponse(tag, b'SEARCH failed: ' + networkString(str(failure.value)))
        log.err(failure)

    def do_FETCH(self, tag, messages, query, uid=0):
        if False:
            while True:
                i = 10
        if query:
            self._oldTimeout = self.setTimeout(None)
            maybeDeferred(self.mbox.fetch, messages, uid=uid).addCallback(iter).addCallback(self.__cbFetch, tag, query, uid).addErrback(self.__ebFetch, tag)
        else:
            self.sendPositiveResponse(tag, b'FETCH complete')
    select_FETCH = (do_FETCH, arg_seqset, arg_fetchatt)

    def __cbFetch(self, results, tag, query, uid):
        if False:
            return 10
        if self.blocked is None:
            self.blocked = []
        try:
            (id, msg) = next(results)
        except StopIteration:
            self.setTimeout(self._oldTimeout)
            del self._oldTimeout
            self.sendPositiveResponse(tag, b'FETCH completed')
            self._unblock()
        else:
            self.spewMessage(id, msg, query, uid).addCallback(lambda _: self.__cbFetch(results, tag, query, uid)).addErrback(self.__ebSpewMessage)

    def __ebSpewMessage(self, failure):
        if False:
            while True:
                i = 10
        log.err(failure)
        self.transport.loseConnection()

    def spew_envelope(self, id, msg, _w=None, _f=None):
        if False:
            while True:
                i = 10
        if _w is None:
            _w = self.transport.write
        _w(b'ENVELOPE ' + collapseNestedLists([getEnvelope(msg)]))

    def spew_flags(self, id, msg, _w=None, _f=None):
        if False:
            return 10
        if _w is None:
            _w = self.transport.writen
        encodedFlags = [networkString(flag) for flag in msg.getFlags()]
        _w(b'FLAGS ' + b'(' + b' '.join(encodedFlags) + b')')

    def spew_internaldate(self, id, msg, _w=None, _f=None):
        if False:
            i = 10
            return i + 15
        if _w is None:
            _w = self.transport.write
        idate = msg.getInternalDate()
        ttup = email.utils.parsedate_tz(nativeString(idate))
        if ttup is None:
            log.msg('%d:%r: unpareseable internaldate: %r' % (id, msg, idate))
            raise IMAP4Exception('Internal failure generating INTERNALDATE')
        strdate = time.strftime('%d-%%s-%Y %H:%M:%S ', ttup[:9])
        odate = networkString(strdate % (_MONTH_NAMES[ttup[1]],))
        if ttup[9] is None:
            odate = odate + b'+0000'
        else:
            if ttup[9] >= 0:
                sign = b'+'
            else:
                sign = b'-'
            odate = odate + sign + b'%04d' % (abs(ttup[9]) // 3600 * 100 + abs(ttup[9]) % 3600 // 60,)
        _w(b'INTERNALDATE ' + _quote(odate))

    def spew_rfc822header(self, id, msg, _w=None, _f=None):
        if False:
            print('Hello World!')
        if _w is None:
            _w = self.transport.write
        hdrs = _formatHeaders(msg.getHeaders(True))
        _w(b'RFC822.HEADER ' + _literal(hdrs))

    def spew_rfc822text(self, id, msg, _w=None, _f=None):
        if False:
            i = 10
            return i + 15
        if _w is None:
            _w = self.transport.write
        _w(b'RFC822.TEXT ')
        _f()
        return FileProducer(msg.getBodyFile()).beginProducing(self.transport)

    def spew_rfc822size(self, id, msg, _w=None, _f=None):
        if False:
            i = 10
            return i + 15
        if _w is None:
            _w = self.transport.write
        _w(b'RFC822.SIZE %d' % (msg.getSize(),))

    def spew_rfc822(self, id, msg, _w=None, _f=None):
        if False:
            i = 10
            return i + 15
        if _w is None:
            _w = self.transport.write
        _w(b'RFC822 ')
        _f()
        mf = IMessageFile(msg, None)
        if mf is not None:
            return FileProducer(mf.open()).beginProducing(self.transport)
        return MessageProducer(msg, None, self._scheduler).beginProducing(self.transport)

    def spew_uid(self, id, msg, _w=None, _f=None):
        if False:
            return 10
        if _w is None:
            _w = self.transport.write
        _w(b'UID %d' % (msg.getUID(),))

    def spew_bodystructure(self, id, msg, _w=None, _f=None):
        if False:
            return 10
        _w(b'BODYSTRUCTURE ' + collapseNestedLists([getBodyStructure(msg, True)]))

    def spew_body(self, part, id, msg, _w=None, _f=None):
        if False:
            for i in range(10):
                print('nop')
        if _w is None:
            _w = self.transport.write
        for p in part.part:
            if msg.isMultipart():
                msg = msg.getSubPart(p)
            elif p > 0:
                raise TypeError('Requested subpart of non-multipart message')
        if part.header:
            hdrs = msg.getHeaders(part.header.negate, *part.header.fields)
            hdrs = _formatHeaders(hdrs)
            _w(part.__bytes__() + b' ' + _literal(hdrs))
        elif part.text:
            _w(part.__bytes__() + b' ')
            _f()
            return FileProducer(msg.getBodyFile()).beginProducing(self.transport)
        elif part.mime:
            hdrs = _formatHeaders(msg.getHeaders(True))
            _w(part.__bytes__() + b' ' + _literal(hdrs))
        elif part.empty:
            _w(part.__bytes__() + b' ')
            _f()
            if part.part:
                return FileProducer(msg.getBodyFile()).beginProducing(self.transport)
            else:
                mf = IMessageFile(msg, None)
                if mf is not None:
                    return FileProducer(mf.open()).beginProducing(self.transport)
                return MessageProducer(msg, None, self._scheduler).beginProducing(self.transport)
        else:
            _w(b'BODY ' + collapseNestedLists([getBodyStructure(msg)]))

    def spewMessage(self, id, msg, query, uid):
        if False:
            print('Hello World!')
        wbuf = WriteBuffer(self.transport)
        write = wbuf.write
        flush = wbuf.flush

        def start():
            if False:
                while True:
                    i = 10
            write(b'* %d FETCH (' % (id,))

        def finish():
            if False:
                while True:
                    i = 10
            write(b')\r\n')

        def space():
            if False:
                for i in range(10):
                    print('nop')
            write(b' ')

        def spew():
            if False:
                while True:
                    i = 10
            seenUID = False
            start()
            for part in query:
                if part.type == 'uid':
                    seenUID = True
                if part.type == 'body':
                    yield self.spew_body(part, id, msg, write, flush)
                else:
                    f = getattr(self, 'spew_' + part.type)
                    yield f(id, msg, write, flush)
                if part is not query[-1]:
                    space()
            if uid and (not seenUID):
                space()
                yield self.spew_uid(id, msg, write, flush)
            finish()
            flush()
        return self._scheduler(spew())

    def __ebFetch(self, failure, tag):
        if False:
            while True:
                i = 10
        self.setTimeout(self._oldTimeout)
        del self._oldTimeout
        log.err(failure)
        self.sendBadResponse(tag, b'FETCH failed: ' + networkString(str(failure.value)))

    def do_STORE(self, tag, messages, mode, flags, uid=0):
        if False:
            print('Hello World!')
        mode = mode.upper()
        silent = mode.endswith(b'SILENT')
        if mode.startswith(b'+'):
            mode = 1
        elif mode.startswith(b'-'):
            mode = -1
        else:
            mode = 0
        flags = [nativeString(flag) for flag in flags]
        maybeDeferred(self.mbox.store, messages, flags, mode, uid=uid).addCallbacks(self.__cbStore, self.__ebStore, (tag, self.mbox, uid, silent), None, (tag,), None)
    select_STORE = (do_STORE, arg_seqset, arg_atom, arg_flaglist)

    def __cbStore(self, result, tag, mbox, uid, silent):
        if False:
            print('Hello World!')
        if result and (not silent):
            for (k, v) in result.items():
                if uid:
                    uidstr = b' UID %d' % (mbox.getUID(k),)
                else:
                    uidstr = b''
                flags = [networkString(flag) for flag in v]
                self.sendUntaggedResponse(b'%d FETCH (FLAGS (%b)%b)' % (k, b' '.join(flags), uidstr))
        self.sendPositiveResponse(tag, b'STORE completed')

    def __ebStore(self, failure, tag):
        if False:
            i = 10
            return i + 15
        self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))

    def do_COPY(self, tag, messages, mailbox, uid=0):
        if False:
            return 10
        mailbox = _parseMbox(mailbox)
        maybeDeferred(self.account.select, mailbox).addCallback(self._cbCopySelectedMailbox, tag, messages, mailbox, uid).addErrback(self._ebCopySelectedMailbox, tag)
    select_COPY = (do_COPY, arg_seqset, arg_finalastring)

    def _cbCopySelectedMailbox(self, mbox, tag, messages, mailbox, uid):
        if False:
            for i in range(10):
                print('nop')
        if not mbox:
            self.sendNegativeResponse(tag, 'No such mailbox: ' + mailbox)
        else:
            maybeDeferred(self.mbox.fetch, messages, uid).addCallback(self.__cbCopy, tag, mbox).addCallback(self.__cbCopied, tag, mbox).addErrback(self.__ebCopy, tag)

    def _ebCopySelectedMailbox(self, failure, tag):
        if False:
            while True:
                i = 10
        self.sendBadResponse(tag, b'Server error: ' + networkString(str(failure.value)))

    def __cbCopy(self, messages, tag, mbox):
        if False:
            return 10
        addedDeferreds = []
        fastCopyMbox = IMessageCopier(mbox, None)
        for (id, msg) in messages:
            if fastCopyMbox is not None:
                d = maybeDeferred(fastCopyMbox.copy, msg)
                addedDeferreds.append(d)
                continue
            flags = msg.getFlags()
            date = msg.getInternalDate()
            body = IMessageFile(msg, None)
            if body is not None:
                bodyFile = body.open()
                d = maybeDeferred(mbox.addMessage, bodyFile, flags, date)
            else:

                def rewind(f):
                    if False:
                        while True:
                            i = 10
                    f.seek(0)
                    return f
                buffer = tempfile.TemporaryFile()
                d = MessageProducer(msg, buffer, self._scheduler).beginProducing(None).addCallback(lambda _, b=buffer, f=flags, d=date: mbox.addMessage(rewind(b), f, d))
            addedDeferreds.append(d)
        return defer.DeferredList(addedDeferreds)

    def __cbCopied(self, deferredIds, tag, mbox):
        if False:
            return 10
        ids = []
        failures = []
        for (status, result) in deferredIds:
            if status:
                ids.append(result)
            else:
                failures.append(result.value)
        if failures:
            self.sendNegativeResponse(tag, '[ALERT] Some messages were not copied')
        else:
            self.sendPositiveResponse(tag, b'COPY completed')

    def __ebCopy(self, failure, tag):
        if False:
            i = 10
            return i + 15
        self.sendBadResponse(tag, b'COPY failed:' + networkString(str(failure.value)))
        log.err(failure)

    def do_UID(self, tag, command, line):
        if False:
            return 10
        command = command.upper()
        if command not in (b'COPY', b'FETCH', b'STORE', b'SEARCH'):
            raise IllegalClientResponse(command)
        self.dispatchCommand(tag, command, line, uid=1)
    select_UID = (do_UID, arg_atom, arg_line)

    def modeChanged(self, writeable):
        if False:
            while True:
                i = 10
        if writeable:
            self.sendUntaggedResponse(message=b'[READ-WRITE]', isAsync=True)
        else:
            self.sendUntaggedResponse(message=b'[READ-ONLY]', isAsync=True)

    def flagsChanged(self, newFlags):
        if False:
            i = 10
            return i + 15
        for (mId, flags) in newFlags.items():
            encodedFlags = [networkString(flag) for flag in flags]
            msg = b'%d FETCH (FLAGS (%b))' % (mId, b' '.join(encodedFlags))
            self.sendUntaggedResponse(msg, isAsync=True)

    def newMessages(self, exists, recent):
        if False:
            return 10
        if exists is not None:
            self.sendUntaggedResponse(b'%d EXISTS' % (exists,), isAsync=True)
        if recent is not None:
            self.sendUntaggedResponse(b'%d RECENT' % (recent,), isAsync=True)
TIMEOUT_ERROR = error.TimeoutError()

@implementer(IMailboxListener)
class IMAP4Client(basic.LineReceiver, policies.TimeoutMixin):
    """IMAP4 client protocol implementation

    @ivar state: A string representing the state the connection is currently
    in.
    """
    tags = None
    waiting = None
    queued = None
    tagID = 1
    state = None
    startedTLS = False
    timeout = 0
    _capCache = None
    _memoryFileLimit = 1024 * 1024 * 10
    authenticators = None
    STATUS_CODES = ('OK', 'NO', 'BAD', 'PREAUTH', 'BYE')
    STATUS_TRANSFORMATIONS = {'MESSAGES': int, 'RECENT': int, 'UNSEEN': int}
    context = None

    def __init__(self, contextFactory=None):
        if False:
            i = 10
            return i + 15
        self.tags = {}
        self.queued = []
        self.authenticators = {}
        self.context = contextFactory
        self._tag = None
        self._parts = None
        self._lastCmd = None

    def registerAuthenticator(self, auth):
        if False:
            while True:
                i = 10
        '\n        Register a new form of authentication\n\n        When invoking the authenticate() method of IMAP4Client, the first\n        matching authentication scheme found will be used.  The ordering is\n        that in which the server lists support authentication schemes.\n\n        @type auth: Implementor of C{IClientAuthentication}\n        @param auth: The object to use to perform the client\n        side of this authentication scheme.\n        '
        self.authenticators[auth.getName().upper()] = auth

    def rawDataReceived(self, data):
        if False:
            i = 10
            return i + 15
        if self.timeout > 0:
            self.resetTimeout()
        self._pendingSize -= len(data)
        if self._pendingSize > 0:
            self._pendingBuffer.write(data)
        else:
            passon = b''
            if self._pendingSize < 0:
                (data, passon) = (data[:self._pendingSize], data[self._pendingSize:])
            self._pendingBuffer.write(data)
            rest = self._pendingBuffer
            self._pendingBuffer = None
            self._pendingSize = None
            rest.seek(0, 0)
            self._parts.append(rest.read())
            self.setLineMode(passon.lstrip(b'\r\n'))

    def _setupForLiteral(self, rest, octets):
        if False:
            i = 10
            return i + 15
        self._pendingBuffer = self.messageFile(octets)
        self._pendingSize = octets
        if self._parts is None:
            self._parts = [rest, b'\r\n']
        else:
            self._parts.extend([rest, b'\r\n'])
        self.setRawMode()

    def connectionMade(self):
        if False:
            while True:
                i = 10
        if self.timeout > 0:
            self.setTimeout(self.timeout)

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        '\n        We are no longer connected\n        '
        if self.timeout > 0:
            self.setTimeout(None)
        if self.queued is not None:
            queued = self.queued
            self.queued = None
            for cmd in queued:
                cmd.defer.errback(reason)
        if self.tags is not None:
            tags = self.tags
            self.tags = None
            for cmd in tags.values():
                if cmd is not None and cmd.defer is not None:
                    cmd.defer.errback(reason)

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Attempt to parse a single line from the server.\n\n        @type line: L{bytes}\n        @param line: The line from the server, without the line delimiter.\n\n        @raise IllegalServerResponse: If the line or some part of the line\n            does not represent an allowed message from the server at this time.\n        '
        if self.timeout > 0:
            self.resetTimeout()
        lastPart = line.rfind(b'{')
        if lastPart != -1:
            lastPart = line[lastPart + 1:]
            if lastPart.endswith(b'}'):
                try:
                    octets = int(lastPart[:-1])
                except ValueError:
                    raise IllegalServerResponse(line)
                if self._parts is None:
                    (self._tag, parts) = line.split(None, 1)
                else:
                    parts = line
                self._setupForLiteral(parts, octets)
                return
        if self._parts is None:
            self._regularDispatch(line)
        else:
            self._parts.append(line)
            (tag, rest) = (self._tag, b''.join(self._parts))
            self._tag = self._parts = None
            self.dispatchCommand(tag, rest)

    def timeoutConnection(self):
        if False:
            while True:
                i = 10
        if self._lastCmd and self._lastCmd.defer is not None:
            (d, self._lastCmd.defer) = (self._lastCmd.defer, None)
            d.errback(TIMEOUT_ERROR)
        if self.queued:
            for cmd in self.queued:
                if cmd.defer is not None:
                    (d, cmd.defer) = (cmd.defer, d)
                    d.errback(TIMEOUT_ERROR)
        self.transport.loseConnection()

    def _regularDispatch(self, line):
        if False:
            for i in range(10):
                print('nop')
        parts = line.split(None, 1)
        if len(parts) != 2:
            parts.append(b'')
        (tag, rest) = parts
        self.dispatchCommand(tag, rest)

    def messageFile(self, octets):
        if False:
            print('Hello World!')
        '\n        Create a file to which an incoming message may be written.\n\n        @type octets: L{int}\n        @param octets: The number of octets which will be written to the file\n\n        @rtype: Any object which implements C{write(string)} and\n        C{seek(int, int)}\n        @return: A file-like object\n        '
        if octets > self._memoryFileLimit:
            return tempfile.TemporaryFile()
        else:
            return BytesIO()

    def makeTag(self):
        if False:
            for i in range(10):
                print('nop')
        tag = ('%0.4X' % self.tagID).encode('ascii')
        self.tagID += 1
        return tag

    def dispatchCommand(self, tag, rest):
        if False:
            print('Hello World!')
        if self.state is None:
            f = self.response_UNAUTH
        else:
            f = getattr(self, 'response_' + self.state.upper(), None)
        if f:
            try:
                f(tag, rest)
            except BaseException:
                log.err()
                self.transport.loseConnection()
        else:
            log.err(f'Cannot dispatch: {self.state}, {tag!r}, {rest!r}')
            self.transport.loseConnection()

    def response_UNAUTH(self, tag, rest):
        if False:
            print('Hello World!')
        if self.state is None:
            (status, rest) = rest.split(None, 1)
            if status.upper() == b'OK':
                self.state = 'unauth'
            elif status.upper() == b'PREAUTH':
                self.state = 'auth'
            else:
                self.transport.loseConnection()
                raise IllegalServerResponse(tag + b' ' + rest)
            (b, e) = (rest.find(b'['), rest.find(b']'))
            if b != -1 and e != -1:
                self.serverGreeting(self.__cbCapabilities(([parseNestedParens(rest[b + 1:e])], None)))
            else:
                self.serverGreeting(None)
        else:
            self._defaultHandler(tag, rest)

    def response_AUTH(self, tag, rest):
        if False:
            for i in range(10):
                print('nop')
        self._defaultHandler(tag, rest)

    def _defaultHandler(self, tag, rest):
        if False:
            print('Hello World!')
        if tag == b'*' or tag == b'+':
            if not self.waiting:
                self._extraInfo([parseNestedParens(rest)])
            else:
                cmd = self.tags[self.waiting]
                if tag == b'+':
                    cmd.continuation(rest)
                else:
                    cmd.lines.append(rest)
        else:
            try:
                cmd = self.tags[tag]
            except KeyError:
                self.transport.loseConnection()
                raise IllegalServerResponse(tag + b' ' + rest)
            else:
                (status, line) = rest.split(None, 1)
                if status == b'OK':
                    cmd.finish(rest, self._extraInfo)
                else:
                    cmd.defer.errback(IMAP4Exception(line))
                del self.tags[tag]
                self.waiting = None
                self._flushQueue()

    def _flushQueue(self):
        if False:
            return 10
        if self.queued:
            cmd = self.queued.pop(0)
            t = self.makeTag()
            self.tags[t] = cmd
            self.sendLine(cmd.format(t))
            self.waiting = t

    def _extraInfo(self, lines):
        if False:
            return 10
        flags = {}
        recent = exists = None
        for response in lines:
            elements = len(response)
            if elements == 1 and response[0] == [b'READ-ONLY']:
                self.modeChanged(False)
            elif elements == 1 and response[0] == [b'READ-WRITE']:
                self.modeChanged(True)
            elif elements == 2 and response[1] == b'EXISTS':
                exists = int(response[0])
            elif elements == 2 and response[1] == b'RECENT':
                recent = int(response[0])
            elif elements == 3 and response[1] == b'FETCH':
                mId = int(response[0])
                (values, _) = self._parseFetchPairs(response[2])
                flags.setdefault(mId, []).extend(values.get('FLAGS', ()))
            else:
                log.msg(f'Unhandled unsolicited response: {response}')
        if flags:
            self.flagsChanged(flags)
        if recent is not None or exists is not None:
            self.newMessages(exists, recent)

    def sendCommand(self, cmd):
        if False:
            print('Hello World!')
        cmd.defer = defer.Deferred()
        if self.waiting:
            self.queued.append(cmd)
            return cmd.defer
        t = self.makeTag()
        self.tags[t] = cmd
        self.sendLine(cmd.format(t))
        self.waiting = t
        self._lastCmd = cmd
        return cmd.defer

    def getCapabilities(self, useCache=1):
        if False:
            return 10
        '\n        Request the capabilities available on this server.\n\n        This command is allowed in any state of connection.\n\n        @type useCache: C{bool}\n        @param useCache: Specify whether to use the capability-cache or to\n        re-retrieve the capabilities from the server.  Server capabilities\n        should never change, so for normal use, this flag should never be\n        false.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback will be invoked with a\n        dictionary mapping capability types to lists of supported\n        mechanisms, or to None if a support list is not applicable.\n        '
        if useCache and self._capCache is not None:
            return defer.succeed(self._capCache)
        cmd = b'CAPABILITY'
        resp = (b'CAPABILITY',)
        d = self.sendCommand(Command(cmd, wantResponse=resp))
        d.addCallback(self.__cbCapabilities)
        return d

    def __cbCapabilities(self, result):
        if False:
            return 10
        (lines, tagline) = result
        caps = {}
        for rest in lines:
            for cap in rest[1:]:
                parts = cap.split(b'=', 1)
                if len(parts) == 1:
                    (category, value) = (parts[0], None)
                else:
                    (category, value) = parts
                caps.setdefault(category, []).append(value)
        for category in caps:
            if caps[category] == [None]:
                caps[category] = None
        self._capCache = caps
        return caps

    def logout(self):
        if False:
            i = 10
            return i + 15
        '\n        Inform the server that we are done with the connection.\n\n        This command is allowed in any state of connection.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback will be invoked with None\n        when the proper server acknowledgement has been received.\n        '
        d = self.sendCommand(Command(b'LOGOUT', wantResponse=(b'BYE',)))
        d.addCallback(self.__cbLogout)
        return d

    def __cbLogout(self, result):
        if False:
            for i in range(10):
                print('nop')
        (lines, tagline) = result
        self.transport.loseConnection()
        return None

    def noop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform no operation.\n\n        This command is allowed in any state of connection.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback will be invoked with a list\n        of untagged status updates the server responds with.\n        '
        d = self.sendCommand(Command(b'NOOP'))
        d.addCallback(self.__cbNoop)
        return d

    def __cbNoop(self, result):
        if False:
            for i in range(10):
                print('nop')
        (lines, tagline) = result
        return lines

    def startTLS(self, contextFactory=None):
        if False:
            while True:
                i = 10
        "\n        Initiates a 'STARTTLS' request and negotiates the TLS / SSL\n        Handshake.\n\n        @param contextFactory: The TLS / SSL Context Factory to\n        leverage.  If the contextFactory is None the IMAP4Client will\n        either use the current TLS / SSL Context Factory or attempt to\n        create a new one.\n\n        @type contextFactory: C{ssl.ClientContextFactory}\n\n        @return: A Deferred which fires when the transport has been\n        secured according to the given contextFactory, or which fails\n        if the transport cannot be secured.\n        "
        assert not self.startedTLS, 'Client and Server are currently communicating via TLS'
        if contextFactory is None:
            contextFactory = self._getContextFactory()
        if contextFactory is None:
            return defer.fail(IMAP4Exception('IMAP4Client requires a TLS context to initiate the STARTTLS handshake'))
        if b'STARTTLS' not in self._capCache:
            return defer.fail(IMAP4Exception('Server does not support secure communication via TLS / SSL'))
        tls = interfaces.ITLSTransport(self.transport, None)
        if tls is None:
            return defer.fail(IMAP4Exception('IMAP4Client transport does not implement interfaces.ITLSTransport'))
        d = self.sendCommand(Command(b'STARTTLS'))
        d.addCallback(self._startedTLS, contextFactory)
        d.addCallback(lambda _: self.getCapabilities())
        return d

    def authenticate(self, secret):
        if False:
            i = 10
            return i + 15
        '\n        Attempt to enter the authenticated state with the server\n\n        This command is allowed in the Non-Authenticated state.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked if the authentication\n        succeeds and whose errback will be invoked otherwise.\n        '
        if self._capCache is None:
            d = self.getCapabilities()
        else:
            d = defer.succeed(self._capCache)
        d.addCallback(self.__cbAuthenticate, secret)
        return d

    def __cbAuthenticate(self, caps, secret):
        if False:
            while True:
                i = 10
        auths = caps.get(b'AUTH', ())
        for scheme in auths:
            if scheme.upper() in self.authenticators:
                cmd = Command(b'AUTHENTICATE', scheme, (), self.__cbContinueAuth, scheme, secret)
                return self.sendCommand(cmd)
        if self.startedTLS:
            return defer.fail(NoSupportedAuthentication(auths, self.authenticators.keys()))
        else:

            def ebStartTLS(err):
                if False:
                    for i in range(10):
                        print('nop')
                err.trap(IMAP4Exception)
                return defer.fail(NoSupportedAuthentication(auths, self.authenticators.keys()))
            d = self.startTLS()
            d.addErrback(ebStartTLS)
            d.addCallback(lambda _: self.getCapabilities())
            d.addCallback(self.__cbAuthTLS, secret)
            return d

    def __cbContinueAuth(self, rest, scheme, secret):
        if False:
            print('Hello World!')
        try:
            chal = decodebytes(rest + b'\n')
        except binascii.Error:
            self.sendLine(b'*')
            raise IllegalServerResponse(rest)
        else:
            auth = self.authenticators[scheme]
            chal = auth.challengeResponse(secret, chal)
            self.sendLine(encodebytes(chal).strip())

    def __cbAuthTLS(self, caps, secret):
        if False:
            return 10
        auths = caps.get(b'AUTH', ())
        for scheme in auths:
            if scheme.upper() in self.authenticators:
                cmd = Command(b'AUTHENTICATE', scheme, (), self.__cbContinueAuth, scheme, secret)
                return self.sendCommand(cmd)
        raise NoSupportedAuthentication(auths, self.authenticators.keys())

    def login(self, username, password):
        if False:
            while True:
                i = 10
        '\n        Authenticate with the server using a username and password\n\n        This command is allowed in the Non-Authenticated state.  If the\n        server supports the STARTTLS capability and our transport supports\n        TLS, TLS is negotiated before the login command is issued.\n\n        A more secure way to log in is to use C{startTLS} or\n        C{authenticate} or both.\n\n        @type username: L{str}\n        @param username: The username to log in with\n\n        @type password: L{str}\n        @param password: The password to log in with\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked if login is successful\n        and whose errback is invoked otherwise.\n        '
        d = maybeDeferred(self.getCapabilities)
        d.addCallback(self.__cbLoginCaps, username, password)
        return d

    def serverGreeting(self, caps):
        if False:
            print('Hello World!')
        '\n        Called when the server has sent us a greeting.\n\n        @type caps: C{dict}\n        @param caps: Capabilities the server advertised in its greeting.\n        '

    def _getContextFactory(self):
        if False:
            return 10
        if self.context is not None:
            return self.context
        try:
            from twisted.internet import ssl
        except ImportError:
            return None
        else:
            return ssl.ClientContextFactory()

    def __cbLoginCaps(self, capabilities, username, password):
        if False:
            for i in range(10):
                print('nop')
        tryTLS = b'STARTTLS' in capabilities
        tlsableTransport = interfaces.ITLSTransport(self.transport, None) is not None
        nontlsTransport = interfaces.ISSLTransport(self.transport, None) is None
        if not self.startedTLS and tryTLS and tlsableTransport and nontlsTransport:
            d = self.startTLS()
            d.addCallbacks(self.__cbLoginTLS, self.__ebLoginTLS, callbackArgs=(username, password))
            return d
        else:
            if nontlsTransport:
                log.msg('Server has no TLS support. logging in over cleartext!')
            args = b' '.join((_quote(username), _quote(password)))
            return self.sendCommand(Command(b'LOGIN', args))

    def _startedTLS(self, result, context):
        if False:
            return 10
        self.transport.startTLS(context)
        self._capCache = None
        self.startedTLS = True
        return result

    def __cbLoginTLS(self, result, username, password):
        if False:
            for i in range(10):
                print('nop')
        args = b' '.join((_quote(username), _quote(password)))
        return self.sendCommand(Command(b'LOGIN', args))

    def __ebLoginTLS(self, failure):
        if False:
            return 10
        log.err(failure)
        return failure

    def namespace(self):
        if False:
            return 10
        "\n        Retrieve information about the namespaces available to this account\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with namespace\n        information.  An example of this information is::\n\n            [[['', '/']], [], []]\n\n        which indicates a single personal namespace called '' with '/'\n        as its hierarchical delimiter, and no shared or user namespaces.\n        "
        cmd = b'NAMESPACE'
        resp = (b'NAMESPACE',)
        d = self.sendCommand(Command(cmd, wantResponse=resp))
        d.addCallback(self.__cbNamespace)
        return d

    def __cbNamespace(self, result):
        if False:
            while True:
                i = 10
        (lines, last) = result

        def _prepareNamespaceOrDelimiter(namespaceList):
            if False:
                print('Hello World!')
            return [element.decode('imap4-utf-7') for element in namespaceList]
        for parts in lines:
            if len(parts) == 4 and parts[0] == b'NAMESPACE':
                return [[] if pairOrNone is None else [_prepareNamespaceOrDelimiter(value) for value in pairOrNone] for pairOrNone in parts[1:]]
        log.err('No NAMESPACE response to NAMESPACE command')
        return [[], [], []]

    def select(self, mailbox):
        if False:
            return 10
        '\n        Select a mailbox\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type mailbox: L{str}\n        @param mailbox: The name of the mailbox to select\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with mailbox\n        information if the select is successful and whose errback is\n        invoked otherwise.  Mailbox information consists of a dictionary\n        with the following L{str} keys and values::\n\n                FLAGS: A list of strings containing the flags settable on\n                        messages in this mailbox.\n\n                EXISTS: An integer indicating the number of messages in this\n                        mailbox.\n\n                RECENT: An integer indicating the number of "recent"\n                        messages in this mailbox.\n\n                UNSEEN: The message sequence number (an integer) of the\n                        first unseen message in the mailbox.\n\n                PERMANENTFLAGS: A list of strings containing the flags that\n                        can be permanently set on messages in this mailbox.\n\n                UIDVALIDITY: An integer uniquely identifying this mailbox.\n        '
        cmd = b'SELECT'
        args = _prepareMailboxName(mailbox)
        resp = ('FLAGS', 'EXISTS', 'RECENT', 'UNSEEN', 'PERMANENTFLAGS', 'UIDVALIDITY')
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbSelect, 1)
        return d

    def examine(self, mailbox):
        if False:
            for i in range(10):
                print('nop')
        '\n        Select a mailbox in read-only mode\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type mailbox: L{str}\n        @param mailbox: The name of the mailbox to examine\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with mailbox\n        information if the examine is successful and whose errback\n        is invoked otherwise.  Mailbox information consists of a dictionary\n        with the following keys and values::\n\n            \'FLAGS\': A list of strings containing the flags settable on\n                        messages in this mailbox.\n\n            \'EXISTS\': An integer indicating the number of messages in this\n                        mailbox.\n\n            \'RECENT\': An integer indicating the number of "recent"\n                        messages in this mailbox.\n\n            \'UNSEEN\': An integer indicating the number of messages not\n                        flagged \\Seen in this mailbox.\n\n            \'PERMANENTFLAGS\': A list of strings containing the flags that\n                        can be permanently set on messages in this mailbox.\n\n            \'UIDVALIDITY\': An integer uniquely identifying this mailbox.\n        '
        cmd = b'EXAMINE'
        args = _prepareMailboxName(mailbox)
        resp = (b'FLAGS', b'EXISTS', b'RECENT', b'UNSEEN', b'PERMANENTFLAGS', b'UIDVALIDITY')
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbSelect, 0)
        return d

    def _intOrRaise(self, value, phrase):
        if False:
            i = 10
            return i + 15
        '\n        Parse C{value} as an integer and return the result or raise\n        L{IllegalServerResponse} with C{phrase} as an argument if C{value}\n        cannot be parsed as an integer.\n        '
        try:
            return int(value)
        except ValueError:
            raise IllegalServerResponse(phrase)

    def __cbSelect(self, result, rw):
        if False:
            return 10
        '\n        Handle lines received in response to a SELECT or EXAMINE command.\n\n        See RFC 3501, section 6.3.1.\n        '
        (lines, tagline) = result
        datum = {'READ-WRITE': rw}
        lines.append(parseNestedParens(tagline))
        for split in lines:
            if len(split) > 0 and split[0].upper() == b'OK':
                content = split[1]
                if isinstance(content, list):
                    key = content[0]
                else:
                    key = content
                key = key.upper()
                if key == b'READ-ONLY':
                    datum['READ-WRITE'] = False
                elif key == b'READ-WRITE':
                    datum['READ-WRITE'] = True
                elif key == b'UIDVALIDITY':
                    datum['UIDVALIDITY'] = self._intOrRaise(content[1], split)
                elif key == b'UNSEEN':
                    datum['UNSEEN'] = self._intOrRaise(content[1], split)
                elif key == b'UIDNEXT':
                    datum['UIDNEXT'] = self._intOrRaise(content[1], split)
                elif key == b'PERMANENTFLAGS':
                    datum['PERMANENTFLAGS'] = tuple((nativeString(flag) for flag in content[1]))
                else:
                    log.err(f'Unhandled SELECT response (2): {split}')
            elif len(split) == 2:
                if split[0].upper() == b'FLAGS':
                    datum['FLAGS'] = tuple((nativeString(flag) for flag in split[1]))
                elif isinstance(split[1], bytes):
                    if split[1].upper() == b'EXISTS':
                        datum['EXISTS'] = self._intOrRaise(split[0], split)
                    elif split[1].upper() == b'RECENT':
                        datum['RECENT'] = self._intOrRaise(split[0], split)
                    else:
                        log.err(f'Unhandled SELECT response (0): {split}')
                else:
                    log.err(f'Unhandled SELECT response (1): {split}')
            else:
                log.err(f'Unhandled SELECT response (4): {split}')
        return datum

    def create(self, name):
        if False:
            print('Hello World!')
        '\n        Create a new mailbox on the server\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type name: L{str}\n        @param name: The name of the mailbox to create.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked if the mailbox creation\n        is successful and whose errback is invoked otherwise.\n        '
        return self.sendCommand(Command(b'CREATE', _prepareMailboxName(name)))

    def delete(self, name):
        if False:
            return 10
        '\n        Delete a mailbox\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type name: L{str}\n        @param name: The name of the mailbox to delete.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose calblack is invoked if the mailbox is\n        deleted successfully and whose errback is invoked otherwise.\n        '
        return self.sendCommand(Command(b'DELETE', _prepareMailboxName(name)))

    def rename(self, oldname, newname):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rename a mailbox\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type oldname: L{str}\n        @param oldname: The current name of the mailbox to rename.\n\n        @type newname: L{str}\n        @param newname: The new name to give the mailbox.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked if the rename is\n        successful and whose errback is invoked otherwise.\n        '
        oldname = _prepareMailboxName(oldname)
        newname = _prepareMailboxName(newname)
        return self.sendCommand(Command(b'RENAME', b' '.join((oldname, newname))))

    def subscribe(self, name):
        if False:
            while True:
                i = 10
        "\n        Add a mailbox to the subscription list\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type name: L{str}\n        @param name: The mailbox to mark as 'active' or 'subscribed'\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked if the subscription\n        is successful and whose errback is invoked otherwise.\n        "
        return self.sendCommand(Command(b'SUBSCRIBE', _prepareMailboxName(name)))

    def unsubscribe(self, name):
        if False:
            return 10
        '\n        Remove a mailbox from the subscription list\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type name: L{str}\n        @param name: The mailbox to unsubscribe\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked if the unsubscription\n        is successful and whose errback is invoked otherwise.\n        '
        return self.sendCommand(Command(b'UNSUBSCRIBE', _prepareMailboxName(name)))

    def list(self, reference, wildcard):
        if False:
            print('Hello World!')
        "\n        List a subset of the available mailboxes\n\n        This command is allowed in the Authenticated and Selected\n        states.\n\n        @type reference: L{str}\n        @param reference: The context in which to interpret\n            C{wildcard}\n\n        @type wildcard: L{str}\n        @param wildcard: The pattern of mailbox names to match,\n            optionally including either or both of the '*' and '%'\n            wildcards.  '*' will match zero or more characters and\n            cross hierarchical boundaries.  '%' will also match zero\n            or more characters, but is limited to a single\n            hierarchical level.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a list of\n            L{tuple}s, the first element of which is a L{tuple} of\n            mailbox flags, the second element of which is the\n            hierarchy delimiter for this mailbox, and the third of\n            which is the mailbox name; if the command is unsuccessful,\n            the deferred's errback is invoked instead.  B{NB}: the\n            delimiter and the mailbox name are L{str}s.\n        "
        cmd = b'LIST'
        args = f'"{reference}" "{wildcard}"'.encode('imap4-utf-7')
        resp = (b'LIST',)
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbList, b'LIST')
        return d

    def lsub(self, reference, wildcard):
        if False:
            print('Hello World!')
        '\n        List a subset of the subscribed available mailboxes\n\n        This command is allowed in the Authenticated and Selected states.\n\n        The parameters and returned object are the same as for the L{list}\n        method, with one slight difference: Only mailboxes which have been\n        subscribed can be included in the resulting list.\n        '
        cmd = b'LSUB'
        encodedReference = reference.encode('ascii')
        encodedWildcard = wildcard.encode('imap4-utf-7')
        args = b''.join([b'"', encodedReference, b'" "', encodedWildcard, b'"'])
        resp = (b'LSUB',)
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbList, b'LSUB')
        return d

    def __cbList(self, result, command):
        if False:
            print('Hello World!')
        (lines, last) = result
        results = []
        for parts in lines:
            if len(parts) == 4 and parts[0] == command:
                parts[1] = tuple((nativeString(flag) for flag in parts[1]))
                parts[2] = parts[2].decode('imap4-utf-7')
                parts[3] = parts[3].decode('imap4-utf-7')
                results.append(tuple(parts[1:]))
        return results
    _statusNames = {name: name.encode('ascii') for name in ('MESSAGES', 'RECENT', 'UIDNEXT', 'UIDVALIDITY', 'UNSEEN')}

    def status(self, mailbox, *names):
        if False:
            while True:
                i = 10
        "\n        Retrieve the status of the given mailbox\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type mailbox: L{str}\n        @param mailbox: The name of the mailbox to query\n\n        @type names: L{bytes}\n        @param names: The status names to query.  These may be any number of:\n            C{'MESSAGES'}, C{'RECENT'}, C{'UIDNEXT'}, C{'UIDVALIDITY'}, and\n            C{'UNSEEN'}.\n\n        @rtype: L{Deferred}\n        @return: A deferred which fires with the status information if the\n            command is successful and whose errback is invoked otherwise.  The\n            status information is in the form of a C{dict}.  Each element of\n            C{names} is a key in the dictionary.  The value for each key is the\n            corresponding response from the server.\n        "
        cmd = b'STATUS'
        preparedMailbox = _prepareMailboxName(mailbox)
        try:
            names = b' '.join((self._statusNames[name] for name in names))
        except KeyError:
            raise ValueError(f'Unknown names: {set(names) - set(self._statusNames)!r}')
        args = b''.join([preparedMailbox, b' (', names, b')'])
        resp = (b'STATUS',)
        d = self.sendCommand(Command(cmd, args, wantResponse=resp))
        d.addCallback(self.__cbStatus)
        return d

    def __cbStatus(self, result):
        if False:
            print('Hello World!')
        (lines, last) = result
        status = {}
        for parts in lines:
            if parts[0] == b'STATUS':
                items = parts[2]
                items = [items[i:i + 2] for i in range(0, len(items), 2)]
                for (k, v) in items:
                    try:
                        status[nativeString(k)] = v
                    except UnicodeDecodeError:
                        raise IllegalServerResponse(repr(items))
        for k in status.keys():
            t = self.STATUS_TRANSFORMATIONS.get(k)
            if t:
                try:
                    status[k] = t(status[k])
                except Exception as e:
                    raise IllegalServerResponse('(' + k + ' ' + status[k] + '): ' + str(e))
        return status

    def append(self, mailbox, message, flags=(), date=None):
        if False:
            while True:
                i = 10
        '\n        Add the given message to the given mailbox.\n\n        This command is allowed in the Authenticated and Selected states.\n\n        @type mailbox: L{str}\n        @param mailbox: The mailbox to which to add this message.\n\n        @type message: Any file-like object opened in B{binary mode}.\n        @param message: The message to add, in RFC822 format.  Newlines\n        in this file should be \\r\\n-style.\n\n        @type flags: Any iterable of L{str}\n        @param flags: The flags to associated with this message.\n\n        @type date: L{str}\n        @param date: The date to associate with this message.  This should\n        be of the format DD-MM-YYYY HH:MM:SS +/-HHMM.  For example, in\n        Eastern Standard Time, on July 1st 2004 at half past 1 PM,\n        "01-07-2004 13:30:00 -0500".\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked when this command\n        succeeds or whose errback is invoked if it fails.\n        '
        message.seek(0, 2)
        L = message.tell()
        message.seek(0, 0)
        if date:
            date = networkString(' "%s"' % nativeString(date))
        else:
            date = b''
        encodedFlags = [networkString(flag) for flag in flags]
        cmd = b'%b (%b)%b {%d}' % (_prepareMailboxName(mailbox), b' '.join(encodedFlags), date, L)
        d = self.sendCommand(Command(b'APPEND', cmd, (), self.__cbContinueAppend, message))
        return d

    def __cbContinueAppend(self, lines, message):
        if False:
            for i in range(10):
                print('nop')
        s = basic.FileSender()
        return s.beginFileTransfer(message, self.transport, None).addCallback(self.__cbFinishAppend)

    def __cbFinishAppend(self, foo):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(b'')

    def check(self):
        if False:
            print('Hello World!')
        '\n        Tell the server to perform a checkpoint\n\n        This command is allowed in the Selected state.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked when this command\n        succeeds or whose errback is invoked if it fails.\n        '
        return self.sendCommand(Command(b'CHECK'))

    def close(self):
        if False:
            print('Hello World!')
        '\n        Return the connection to the Authenticated state.\n\n        This command is allowed in the Selected state.\n\n        Issuing this command will also remove all messages flagged \\Deleted\n        from the selected mailbox if it is opened in read-write mode,\n        otherwise it indicates success by no messages are removed.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked when the command\n        completes successfully or whose errback is invoked if it fails.\n        '
        return self.sendCommand(Command(b'CLOSE'))

    def expunge(self):
        if False:
            i = 10
            return i + 15
        "\n        Return the connection to the Authenticate state.\n\n        This command is allowed in the Selected state.\n\n        Issuing this command will perform the same actions as issuing the\n        close command, but will also generate an 'expunge' response for\n        every message deleted.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a list of the\n        'expunge' responses when this command is successful or whose errback\n        is invoked otherwise.\n        "
        cmd = b'EXPUNGE'
        resp = (b'EXPUNGE',)
        d = self.sendCommand(Command(cmd, wantResponse=resp))
        d.addCallback(self.__cbExpunge)
        return d

    def __cbExpunge(self, result):
        if False:
            for i in range(10):
                print('nop')
        (lines, last) = result
        ids = []
        for parts in lines:
            if len(parts) == 2 and parts[1] == b'EXPUNGE':
                ids.append(self._intOrRaise(parts[0], parts))
        return ids

    def search(self, *queries, uid=False):
        if False:
            while True:
                i = 10
        '\n        Search messages in the currently selected mailbox\n\n        This command is allowed in the Selected state.\n\n        Any non-zero number of queries are accepted by this method, as returned\n        by the C{Query}, C{Or}, and C{Not} functions.\n\n        @param uid: if true, the server is asked to return message UIDs instead\n            of message sequence numbers.\n        @type uid: L{bool}\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback will be invoked with a list of all\n            the message sequence numbers return by the search, or whose errback\n            will be invoked if there is an error.\n        '
        queries = [query.encode('charmap') for query in queries]
        cmd = b'UID SEARCH' if uid else b'SEARCH'
        args = b' '.join(queries)
        d = self.sendCommand(Command(cmd, args, wantResponse=(cmd,)))
        d.addCallback(self.__cbSearch)
        return d

    def __cbSearch(self, result):
        if False:
            i = 10
            return i + 15
        (lines, end) = result
        ids = []
        for parts in lines:
            if len(parts) > 0 and parts[0] == b'SEARCH':
                ids.extend([self._intOrRaise(p, parts) for p in parts[1:]])
        return ids

    def fetchUID(self, messages, uid=0):
        if False:
            print('Hello World!')
        '\n        Retrieve the unique identifier for one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message sequence numbers to unique message identifiers, or whose\n        errback is invoked if there is an error.\n        '
        return self._fetch(messages, useUID=uid, uid=1)

    def fetchFlags(self, messages, uid=0):
        if False:
            while True:
                i = 10
        '\n        Retrieve the flags for one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: The messages for which to retrieve flags.\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to lists of flags, or whose errback is invoked if\n        there is an error.\n        '
        return self._fetch(messages, useUID=uid, flags=1)

    def fetchInternalDate(self, messages, uid=0):
        if False:
            while True:
                i = 10
        '\n        Retrieve the internal date associated with one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: The messages for which to retrieve the internal date.\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to date strings, or whose errback is invoked\n        if there is an error.  Date strings take the format of\n        "day-month-year time timezone".\n        '
        return self._fetch(messages, useUID=uid, internaldate=1)

    def fetchEnvelope(self, messages, uid=0):
        if False:
            print('Hello World!')
        '\n        Retrieve the envelope data for one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: The messages for which to retrieve envelope\n            data.\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of\n            message numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict\n            mapping message numbers to envelope data, or whose errback\n            is invoked if there is an error.  Envelope data consists\n            of a sequence of the date, subject, from, sender,\n            reply-to, to, cc, bcc, in-reply-to, and message-id header\n            fields.  The date, subject, in-reply-to, and message-id\n            fields are L{str}, while the from, sender, reply-to, to,\n            cc, and bcc fields contain address data as L{str}s.\n            Address data consists of a sequence of name, source route,\n            mailbox name, and hostname.  Fields which are not present\n            for a particular address may be L{None}.\n        '
        return self._fetch(messages, useUID=uid, envelope=1)

    def fetchBodyStructure(self, messages, uid=0):
        if False:
            return 10
        "\n        Retrieve the structure of the body of one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: The messages for which to retrieve body structure\n        data.\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to body structure data, or whose errback is invoked\n        if there is an error.  Body structure data describes the MIME-IMB\n        format of a message and consists of a sequence of mime type, mime\n        subtype, parameters, content id, description, encoding, and size.\n        The fields following the size field are variable: if the mime\n        type/subtype is message/rfc822, the contained message's envelope\n        information, body structure data, and number of lines of text; if\n        the mime type is text, the number of lines of text.  Extension fields\n        may also be included; if present, they are: the MD5 hash of the body,\n        body disposition, body language.\n        "
        return self._fetch(messages, useUID=uid, bodystructure=1)

    def fetchSimplifiedBody(self, messages, uid=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve the simplified body structure of one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: C{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to body data, or whose errback is invoked\n        if there is an error.  The simplified body structure is the same\n        as the body structure, except that extension fields will never be\n        present.\n        '
        return self._fetch(messages, useUID=uid, body=1)

    def fetchMessage(self, messages, uid=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        Retrieve one or more entire messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: C{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n\n        @return: A L{Deferred} which will fire with a C{dict} mapping message\n            sequence numbers to C{dict}s giving message data for the\n            corresponding message.  If C{uid} is true, the inner dictionaries\n            have a C{'UID'} key mapped to a L{str} giving the UID for the\n            message.  The text of the message is a L{str} associated with the\n            C{'RFC822'} key in each dictionary.\n        "
        return self._fetch(messages, useUID=uid, rfc822=1)

    def fetchHeaders(self, messages, uid=0):
        if False:
            while True:
                i = 10
        '\n        Retrieve headers of one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to dicts of message headers, or whose errback is\n        invoked if there is an error.\n        '
        return self._fetch(messages, useUID=uid, rfc822header=1)

    def fetchBody(self, messages, uid=0):
        if False:
            while True:
                i = 10
        '\n        Retrieve body text of one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to file-like objects containing body text, or whose\n        errback is invoked if there is an error.\n        '
        return self._fetch(messages, useUID=uid, rfc822text=1)

    def fetchSize(self, messages, uid=0):
        if False:
            return 10
        '\n        Retrieve the size, in octets, of one or more messages\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to sizes, or whose errback is invoked if there is\n        an error.\n        '
        return self._fetch(messages, useUID=uid, rfc822size=1)

    def fetchFull(self, messages, uid=0):
        if False:
            while True:
                i = 10
        '\n        Retrieve several different fields of one or more messages\n\n        This command is allowed in the Selected state.  This is equivalent\n        to issuing all of the C{fetchFlags}, C{fetchInternalDate},\n        C{fetchSize}, C{fetchEnvelope}, and C{fetchSimplifiedBody}\n        functions.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to dict of the retrieved data values, or whose\n        errback is invoked if there is an error.  They dictionary keys\n        are "flags", "date", "size", "envelope", and "body".\n        '
        return self._fetch(messages, useUID=uid, flags=1, internaldate=1, rfc822size=1, envelope=1, body=1)

    def fetchAll(self, messages, uid=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve several different fields of one or more messages\n\n        This command is allowed in the Selected state.  This is equivalent\n        to issuing all of the C{fetchFlags}, C{fetchInternalDate},\n        C{fetchSize}, and C{fetchEnvelope} functions.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to dict of the retrieved data values, or whose\n        errback is invoked if there is an error.  They dictionary keys\n        are "flags", "date", "size", and "envelope".\n        '
        return self._fetch(messages, useUID=uid, flags=1, internaldate=1, rfc822size=1, envelope=1)

    def fetchFast(self, messages, uid=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve several different fields of one or more messages\n\n        This command is allowed in the Selected state.  This is equivalent\n        to issuing all of the C{fetchFlags}, C{fetchInternalDate}, and\n        C{fetchSize} functions.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a dict mapping\n        message numbers to dict of the retrieved data values, or whose\n        errback is invoked if there is an error.  They dictionary keys are\n        "flags", "date", and "size".\n        '
        return self._fetch(messages, useUID=uid, flags=1, internaldate=1, rfc822size=1)

    def _parseFetchPairs(self, fetchResponseList):
        if False:
            print('Hello World!')
        '\n        Given the result of parsing a single I{FETCH} response, construct a\n        L{dict} mapping response keys to response values.\n\n        @param fetchResponseList: The result of parsing a I{FETCH} response\n            with L{parseNestedParens} and extracting just the response data\n            (that is, just the part that comes after C{"FETCH"}).  The form\n            of this input (and therefore the output of this method) is very\n            disagreeable.  A valuable improvement would be to enumerate the\n            possible keys (representing them as structured objects of some\n            sort) rather than using strings and tuples of tuples of strings\n            and so forth.  This would allow the keys to be documented more\n            easily and would allow for a much simpler application-facing API\n            (one not based on looking up somewhat hard to predict keys in a\n            dict).  Since C{fetchResponseList} notionally represents a\n            flattened sequence of pairs (identifying keys followed by their\n            associated values), collapsing such complex elements of this\n            list as C{["BODY", ["HEADER.FIELDS", ["SUBJECT"]]]} into a\n            single object would also greatly simplify the implementation of\n            this method.\n\n        @return: A C{dict} of the response data represented by C{pairs}.  Keys\n            in this dictionary are things like C{"RFC822.TEXT"}, C{"FLAGS"}, or\n            C{("BODY", ("HEADER.FIELDS", ("SUBJECT",)))}.  Values are entirely\n            dependent on the key with which they are associated, but retain the\n            same structured as produced by L{parseNestedParens}.\n        '

        def nativeStringResponse(thing):
            if False:
                print('Hello World!')
            if isinstance(thing, bytes):
                return thing.decode('charmap')
            elif isinstance(thing, list):
                return [nativeStringResponse(subthing) for subthing in thing]
        values = {}
        unstructured = []
        responseParts = iter(fetchResponseList)
        while True:
            try:
                key = next(responseParts)
            except StopIteration:
                break
            try:
                value = next(responseParts)
            except StopIteration:
                raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
            if key not in (b'BODY', b'BODY.PEEK'):
                hasSection = False
            elif not isinstance(value, list):
                hasSection = False
            elif len(value) > 2:
                hasSection = False
            elif value and isinstance(value[0], list):
                hasSection = False
            else:
                hasSection = True
            key = nativeString(key)
            unstructured.append(key)
            if hasSection:
                if len(value) < 2:
                    value = [nativeString(v) for v in value]
                    unstructured.append(value)
                    key = (key, tuple(value))
                else:
                    valueHead = nativeString(value[0])
                    valueTail = [nativeString(v) for v in value[1]]
                    unstructured.append([valueHead, valueTail])
                    key = (key, (valueHead, tuple(valueTail)))
                try:
                    value = next(responseParts)
                except StopIteration:
                    raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
                if value.startswith(b'<') and value.endswith(b'>'):
                    try:
                        int(value[1:-1])
                    except ValueError:
                        pass
                    else:
                        value = nativeString(value)
                        unstructured.append(value)
                        key = key + (value,)
                        try:
                            value = next(responseParts)
                        except StopIteration:
                            raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
            value = nativeStringResponse(value)
            unstructured.append(value)
            values[key] = value
        return (values, unstructured)

    def _cbFetch(self, result, requestedParts, structured):
        if False:
            print('Hello World!')
        (lines, last) = result
        info = {}
        for parts in lines:
            if len(parts) == 3 and parts[1] == b'FETCH':
                id = self._intOrRaise(parts[0], parts)
                if id not in info:
                    info[id] = [parts[2]]
                else:
                    info[id][0].extend(parts[2])
        results = {}
        decodedInfo = {}
        for (messageId, values) in info.items():
            (structuredMap, unstructuredList) = self._parseFetchPairs(values[0])
            decodedInfo.setdefault(messageId, [[]])[0].extend(unstructuredList)
            results.setdefault(messageId, {}).update(structuredMap)
        info = decodedInfo
        flagChanges = {}
        for messageId in list(results.keys()):
            values = results[messageId]
            for part in list(values.keys()):
                if part not in requestedParts and part == 'FLAGS':
                    flagChanges[messageId] = values['FLAGS']
                    for i in range(len(info[messageId][0])):
                        if info[messageId][0][i] == 'FLAGS':
                            del info[messageId][0][i:i + 2]
                            break
                    del values['FLAGS']
                    if not values:
                        del results[messageId]
        if flagChanges:
            self.flagsChanged(flagChanges)
        if structured:
            return results
        else:
            return info

    def fetchSpecific(self, messages, uid=0, headerType=None, headerNumber=None, headerArgs=None, peek=None, offset=None, length=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve a specific section of one or more messages\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n            numbers or of unique message IDs.\n\n        @type headerType: L{str}\n        @param headerType: If specified, must be one of HEADER, HEADER.FIELDS,\n            HEADER.FIELDS.NOT, MIME, or TEXT, and will determine which part of\n            the message is retrieved.  For HEADER.FIELDS and HEADER.FIELDS.NOT,\n            C{headerArgs} must be a sequence of header names.  For MIME,\n            C{headerNumber} must be specified.\n\n        @type headerNumber: L{int} or L{int} sequence\n        @param headerNumber: The nested rfc822 index specifying the entity to\n            retrieve.  For example, C{1} retrieves the first entity of the\n            message, and C{(2, 1, 3}) retrieves the 3rd entity inside the first\n            entity inside the second entity of the message.\n\n        @type headerArgs: A sequence of L{str}\n        @param headerArgs: If C{headerType} is HEADER.FIELDS, these are the\n            headers to retrieve.  If it is HEADER.FIELDS.NOT, these are the\n            headers to exclude from retrieval.\n\n        @type peek: C{bool}\n        @param peek: If true, cause the server to not set the \\Seen flag on\n            this message as a result of this command.\n\n        @type offset: L{int}\n        @param offset: The number of octets at the beginning of the result to\n            skip.\n\n        @type length: L{int}\n        @param length: The number of octets to retrieve.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a mapping of message\n            numbers to retrieved data, or whose errback is invoked if there is\n            an error.\n        '
        fmt = '%s BODY%s[%s%s%s]%s'
        if headerNumber is None:
            number = ''
        elif isinstance(headerNumber, int):
            number = str(headerNumber)
        else:
            number = '.'.join(map(str, headerNumber))
        if headerType is None:
            header = ''
        elif number:
            header = '.' + headerType
        else:
            header = headerType
        if header and headerType in ('HEADER.FIELDS', 'HEADER.FIELDS.NOT'):
            if headerArgs is not None:
                payload = ' (%s)' % ' '.join(headerArgs)
            else:
                payload = ' ()'
        else:
            payload = ''
        if offset is None:
            extra = ''
        else:
            extra = '<%d.%d>' % (offset, length)
        fetch = uid and b'UID FETCH' or b'FETCH'
        cmd = fmt % (messages, peek and '.PEEK' or '', number, header, payload, extra)
        cmd = cmd.encode('charmap')
        d = self.sendCommand(Command(fetch, cmd, wantResponse=(b'FETCH',)))
        d.addCallback(self._cbFetch, (), False)
        return d

    def _fetch(self, messages, useUID=0, **terms):
        if False:
            i = 10
            return i + 15
        messages = str(messages).encode('ascii')
        fetch = useUID and b'UID FETCH' or b'FETCH'
        if 'rfc822text' in terms:
            del terms['rfc822text']
            terms['rfc822.text'] = True
        if 'rfc822size' in terms:
            del terms['rfc822size']
            terms['rfc822.size'] = True
        if 'rfc822header' in terms:
            del terms['rfc822header']
            terms['rfc822.header'] = True
        encodedTerms = [networkString(s) for s in terms]
        cmd = messages + b' (' + b' '.join([s.upper() for s in encodedTerms]) + b')'
        d = self.sendCommand(Command(fetch, cmd, wantResponse=(b'FETCH',)))
        d.addCallback(self._cbFetch, [t.upper() for t in terms.keys()], True)
        return d

    def setFlags(self, messages, flags, silent=1, uid=0):
        if False:
            i = 10
            return i + 15
        "\n        Set the flags for one or more messages.\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type flags: Any iterable of L{str}\n        @param flags: The flags to set\n\n        @type silent: L{bool}\n        @param silent: If true, cause the server to suppress its verbose\n        response.\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a list of the\n        server's responses (C{[]} if C{silent} is true) or whose\n        errback is invoked if there is an error.\n        "
        return self._store(messages, b'FLAGS', silent, flags, uid)

    def addFlags(self, messages, flags, silent=1, uid=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        Add to the set flags for one or more messages.\n\n        This command is allowed in the Selected state.\n\n        @type messages: C{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type flags: Any iterable of L{str}\n        @param flags: The flags to set\n\n        @type silent: C{bool}\n        @param silent: If true, cause the server to suppress its verbose\n        response.\n\n        @type uid: C{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a list of the\n        server's responses (C{[]} if C{silent} is true) or whose\n        errback is invoked if there is an error.\n        "
        return self._store(messages, b'+FLAGS', silent, flags, uid)

    def removeFlags(self, messages, flags, silent=1, uid=0):
        if False:
            print('Hello World!')
        "\n        Remove from the set flags for one or more messages.\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type flags: Any iterable of L{str}\n        @param flags: The flags to set\n\n        @type silent: L{bool}\n        @param silent: If true, cause the server to suppress its verbose\n        response.\n\n        @type uid: L{bool}\n        @param uid: Indicates whether the message sequence set is of message\n        numbers or of unique message IDs.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a list of the\n        server's responses (C{[]} if C{silent} is true) or whose\n        errback is invoked if there is an error.\n        "
        return self._store(messages, b'-FLAGS', silent, flags, uid)

    def _store(self, messages, cmd, silent, flags, uid):
        if False:
            print('Hello World!')
        messages = str(messages).encode('ascii')
        encodedFlags = [networkString(flag) for flag in flags]
        if silent:
            cmd = cmd + b'.SILENT'
        store = uid and b'UID STORE' or b'STORE'
        args = b' '.join((messages, cmd, b'(' + b' '.join(encodedFlags) + b')'))
        d = self.sendCommand(Command(store, args, wantResponse=(b'FETCH',)))
        expected = ()
        if not silent:
            expected = ('FLAGS',)
        d.addCallback(self._cbFetch, expected, True)
        return d

    def copy(self, messages, mailbox, uid):
        if False:
            i = 10
            return i + 15
        '\n        Copy the specified messages to the specified mailbox.\n\n        This command is allowed in the Selected state.\n\n        @type messages: L{MessageSet} or L{str}\n        @param messages: A message sequence set\n\n        @type mailbox: L{str}\n        @param mailbox: The mailbox to which to copy the messages\n\n        @type uid: C{bool}\n        @param uid: If true, the C{messages} refers to message UIDs, rather\n        than message sequence numbers.\n\n        @rtype: L{Deferred}\n        @return: A deferred whose callback is invoked with a true value\n        when the copy is successful, or whose errback is invoked if there\n        is an error.\n        '
        messages = str(messages).encode('ascii')
        if uid:
            cmd = b'UID COPY'
        else:
            cmd = b'COPY'
        args = b' '.join([messages, _prepareMailboxName(mailbox)])
        return self.sendCommand(Command(cmd, args))

    def modeChanged(self, writeable):
        if False:
            return 10
        'Override me'

    def flagsChanged(self, newFlags):
        if False:
            print('Hello World!')
        'Override me'

    def newMessages(self, exists, recent):
        if False:
            while True:
                i = 10
        'Override me'

def parseIdList(s, lastMessageId=None):
    if False:
        while True:
            i = 10
    '\n    Parse a message set search key into a C{MessageSet}.\n\n    @type s: L{bytes}\n    @param s: A string description of an id list, for example "1:3, 4:*"\n\n    @type lastMessageId: L{int}\n    @param lastMessageId: The last message sequence id or UID, depending on\n        whether we are parsing the list in UID or sequence id context. The\n        caller should pass in the correct value.\n\n    @rtype: C{MessageSet}\n    @return: A C{MessageSet} that contains the ids defined in the list\n    '
    res = MessageSet()
    parts = s.split(b',')
    for p in parts:
        if b':' in p:
            (low, high) = p.split(b':', 1)
            try:
                if low == b'*':
                    low = None
                else:
                    low = int(low)
                if high == b'*':
                    high = None
                else:
                    high = int(high)
                if low is high is None:
                    raise IllegalIdentifierError(p)
                if low is not None and low <= 0 or (high is not None and high <= 0):
                    raise IllegalIdentifierError(p)
                high = high or lastMessageId
                low = low or lastMessageId
                res.add(low, high)
            except ValueError:
                raise IllegalIdentifierError(p)
        else:
            try:
                if p == b'*':
                    p = None
                else:
                    p = int(p)
                if p is not None and p <= 0:
                    raise IllegalIdentifierError(p)
            except ValueError:
                raise IllegalIdentifierError(p)
            else:
                res.extend(p or lastMessageId)
    return res
_SIMPLE_BOOL = ('ALL', 'ANSWERED', 'DELETED', 'DRAFT', 'FLAGGED', 'NEW', 'OLD', 'RECENT', 'SEEN', 'UNANSWERED', 'UNDELETED', 'UNDRAFT', 'UNFLAGGED', 'UNSEEN')
_NO_QUOTES = ('LARGER', 'SMALLER', 'UID')
_sorted = sorted

def Query(sorted=0, **kwarg):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a query string\n\n    Among the accepted keywords are::\n\n        all         : If set to a true value, search all messages in the\n                      current mailbox\n\n        answered    : If set to a true value, search messages flagged with\n                      \\Answered\n\n        bcc         : A substring to search the BCC header field for\n\n        before      : Search messages with an internal date before this\n                      value.  The given date should be a string in the format\n                      of 'DD-Mon-YYYY'.  For example, '03-Mar-2003'.\n\n        body        : A substring to search the body of the messages for\n\n        cc          : A substring to search the CC header field for\n\n        deleted     : If set to a true value, search messages flagged with\n                      \\Deleted\n\n        draft       : If set to a true value, search messages flagged with\n                      \\Draft\n\n        flagged     : If set to a true value, search messages flagged with\n                      \\Flagged\n\n        from        : A substring to search the From header field for\n\n        header      : A two-tuple of a header name and substring to search\n                      for in that header\n\n        keyword     : Search for messages with the given keyword set\n\n        larger      : Search for messages larger than this number of octets\n\n        messages    : Search only the given message sequence set.\n\n        new         : If set to a true value, search messages flagged with\n                      \\Recent but not \\Seen\n\n        old         : If set to a true value, search messages not flagged with\n                      \\Recent\n\n        on          : Search messages with an internal date which is on this\n                      date.  The given date should be a string in the format\n                      of 'DD-Mon-YYYY'.  For example, '03-Mar-2003'.\n\n        recent      : If set to a true value, search for messages flagged with\n                      \\Recent\n\n        seen        : If set to a true value, search for messages flagged with\n                      \\Seen\n\n        sentbefore  : Search for messages with an RFC822 'Date' header before\n                      this date.  The given date should be a string in the format\n                      of 'DD-Mon-YYYY'.  For example, '03-Mar-2003'.\n\n        senton      : Search for messages with an RFC822 'Date' header which is\n                      on this date  The given date should be a string in the format\n                      of 'DD-Mon-YYYY'.  For example, '03-Mar-2003'.\n\n        sentsince   : Search for messages with an RFC822 'Date' header which is\n                      after this date.  The given date should be a string in the format\n                      of 'DD-Mon-YYYY'.  For example, '03-Mar-2003'.\n\n        since       : Search for messages with an internal date that is after\n                      this date..  The given date should be a string in the format\n                      of 'DD-Mon-YYYY'.  For example, '03-Mar-2003'.\n\n        smaller     : Search for messages smaller than this number of octets\n\n        subject     : A substring to search the 'subject' header for\n\n        text        : A substring to search the entire message for\n\n        to          : A substring to search the 'to' header for\n\n        uid         : Search only the messages in the given message set\n\n        unanswered  : If set to a true value, search for messages not\n                      flagged with \\Answered\n\n        undeleted   : If set to a true value, search for messages not\n                      flagged with \\Deleted\n\n        undraft     : If set to a true value, search for messages not\n                      flagged with \\Draft\n\n        unflagged   : If set to a true value, search for messages not\n                      flagged with \\Flagged\n\n        unkeyword   : Search for messages without the given keyword set\n\n        unseen      : If set to a true value, search for messages not\n                      flagged with \\Seen\n\n    @type sorted: C{bool}\n    @param sorted: If true, the output will be sorted, alphabetically.\n    The standard does not require it, but it makes testing this function\n    easier.  The default is zero, and this should be acceptable for any\n    application.\n\n    @rtype: L{str}\n    @return: The formatted query string\n    "
    cmd = []
    keys = kwarg.keys()
    if sorted:
        keys = _sorted(keys)
    for k in keys:
        v = kwarg[k]
        k = k.upper()
        if k in _SIMPLE_BOOL and v:
            cmd.append(k)
        elif k == 'HEADER':
            cmd.extend([k, str(v[0]), str(v[1])])
        elif k == 'KEYWORD' or k == 'UNKEYWORD':
            v = _nonAtomRE.sub('', v)
            cmd.extend([k, v])
        elif k not in _NO_QUOTES:
            if isinstance(v, MessageSet):
                fmt = '"%s"'
            elif isinstance(v, str):
                fmt = '"%s"'
            else:
                fmt = '"%d"'
            cmd.extend([k, fmt % (v,)])
        elif isinstance(v, int):
            cmd.extend([k, '%d' % (v,)])
        else:
            cmd.extend([k, f'{v}'])
    if len(cmd) > 1:
        return '(' + ' '.join(cmd) + ')'
    else:
        return ' '.join(cmd)

def Or(*args):
    if False:
        print('Hello World!')
    '\n    The disjunction of two or more queries\n    '
    if len(args) < 2:
        raise IllegalQueryError(args)
    elif len(args) == 2:
        return '(OR %s %s)' % args
    else:
        return f'(OR {args[0]} {Or(*args[1:])})'

def Not(query):
    if False:
        return 10
    'The negation of a query'
    return f'(NOT {query})'

def wildcardToRegexp(wildcard, delim=None):
    if False:
        print('Hello World!')
    wildcard = wildcard.replace('*', '(?:.*?)')
    if delim is None:
        wildcard = wildcard.replace('%', '(?:.*?)')
    else:
        wildcard = wildcard.replace('%', '(?:(?:[^%s])*?)' % re.escape(delim))
    return re.compile(wildcard, re.I)

def splitQuoted(s):
    if False:
        while True:
            i = 10
    '\n    Split a string into whitespace delimited tokens\n\n    Tokens that would otherwise be separated but are surrounded by "\n    remain as a single token.  Any token that is not quoted and is\n    equal to "NIL" is tokenized as L{None}.\n\n    @type s: L{bytes}\n    @param s: The string to be split\n\n    @rtype: L{list} of L{bytes}\n    @return: A list of the resulting tokens\n\n    @raise MismatchedQuoting: Raised if an odd number of quotes are present\n    '
    s = s.strip()
    result = []
    word = []
    inQuote = inWord = False
    qu = _matchingString('"', s)
    esc = _matchingString('\\', s)
    empty = _matchingString('', s)
    nil = _matchingString('NIL', s)
    for (i, c) in enumerate(iterbytes(s)):
        if c == qu:
            if i and s[i - 1:i] == esc:
                word.pop()
                word.append(qu)
            elif not inQuote:
                inQuote = True
            else:
                inQuote = False
                result.append(empty.join(word))
                word = []
        elif not inWord and (not inQuote) and (c not in qu + string.whitespace.encode('ascii')):
            inWord = True
            word.append(c)
        elif inWord and (not inQuote) and (c in string.whitespace.encode('ascii')):
            w = empty.join(word)
            if w == nil:
                result.append(None)
            else:
                result.append(w)
            word = []
            inWord = False
        elif inWord or inQuote:
            word.append(c)
    if inQuote:
        raise MismatchedQuoting(s)
    if inWord:
        w = empty.join(word)
        if w == nil:
            result.append(None)
        else:
            result.append(w)
    return result

def splitOn(sequence, predicate, transformers):
    if False:
        print('Hello World!')
    result = []
    mode = predicate(sequence[0])
    tmp = [sequence[0]]
    for e in sequence[1:]:
        p = predicate(e)
        if p != mode:
            result.extend(transformers[mode](tmp))
            tmp = [e]
            mode = p
        else:
            tmp.append(e)
    result.extend(transformers[mode](tmp))
    return result

def collapseStrings(results):
    if False:
        return 10
    "\n    Turns a list of length-one strings and lists into a list of longer\n    strings and lists.  For example,\n\n    ['a', 'b', ['c', 'd']] is returned as ['ab', ['cd']]\n\n    @type results: L{list} of L{bytes} and L{list}\n    @param results: The list to be collapsed\n\n    @rtype: L{list} of L{bytes} and L{list}\n    @return: A new list which is the collapsed form of C{results}\n    "
    copy = []
    begun = None
    pred = lambda e: isinstance(e, tuple)
    tran = {0: lambda e: splitQuoted(b''.join(e)), 1: lambda e: [b''.join([i[0] for i in e])]}
    for (i, c) in enumerate(results):
        if isinstance(c, list):
            if begun is not None:
                copy.extend(splitOn(results[begun:i], pred, tran))
                begun = None
            copy.append(collapseStrings(c))
        elif begun is None:
            begun = i
    if begun is not None:
        copy.extend(splitOn(results[begun:], pred, tran))
    return copy

def parseNestedParens(s, handleLiteral=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse an s-exp-like string into a more useful data structure.\n\n    @type s: L{bytes}\n    @param s: The s-exp-like string to parse\n\n    @rtype: L{list} of L{bytes} and L{list}\n    @return: A list containing the tokens present in the input.\n\n    @raise MismatchedNesting: Raised if the number or placement\n    of opening or closing parenthesis is invalid.\n    '
    s = s.strip()
    inQuote = 0
    contentStack = [[]]
    try:
        i = 0
        L = len(s)
        while i < L:
            c = s[i:i + 1]
            if inQuote:
                if c == b'\\':
                    contentStack[-1].append(s[i:i + 2])
                    i += 2
                    continue
                elif c == b'"':
                    inQuote = not inQuote
                contentStack[-1].append(c)
                i += 1
            elif c == b'"':
                contentStack[-1].append(c)
                inQuote = not inQuote
                i += 1
            elif handleLiteral and c == b'{':
                end = s.find(b'}', i)
                if end == -1:
                    raise ValueError('Malformed literal')
                literalSize = int(s[i + 1:end])
                contentStack[-1].append((s[end + 3:end + 3 + literalSize],))
                i = end + 3 + literalSize
            elif c == b'(' or c == b'[':
                contentStack.append([])
                i += 1
            elif c == b')' or c == b']':
                contentStack[-2].append(contentStack.pop())
                i += 1
            else:
                contentStack[-1].append(c)
                i += 1
    except IndexError:
        raise MismatchedNesting(s)
    if len(contentStack) != 1:
        raise MismatchedNesting(s)
    return collapseStrings(contentStack[0])

def _quote(s):
    if False:
        return 10
    qu = _matchingString('"', s)
    esc = _matchingString('\\', s)
    return qu + s.replace(esc, esc + esc).replace(qu, esc + qu) + qu

def _literal(s: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    return b'{%d}\r\n%b' % (len(s), s)

class DontQuoteMe:

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(self.value)
_ATOM_SPECIALS = b'(){ %*"'

def _needsQuote(s):
    if False:
        print('Hello World!')
    if s == b'':
        return 1
    for c in iterbytes(s):
        if c < b' ' or c > b'\x7f':
            return 1
        if c in _ATOM_SPECIALS:
            return 1
    return 0

def _parseMbox(name):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(name, str):
        return name
    try:
        return name.decode('imap4-utf-7')
    except BaseException:
        log.err()
        raise IllegalMailboxEncoding(name)

def _prepareMailboxName(name):
    if False:
        while True:
            i = 10
    if not isinstance(name, str):
        name = name.decode('charmap')
    name = name.encode('imap4-utf-7')
    if _needsQuote(name):
        return _quote(name)
    return name

def _needsLiteral(s):
    if False:
        i = 10
        return i + 15
    cr = _matchingString('\n', s)
    lf = _matchingString('\r', s)
    return cr in s or lf in s or len(s) > 1000

def collapseNestedLists(items):
    if False:
        i = 10
        return i + 15
    "\n    Turn a nested list structure into an s-exp-like string.\n\n    Strings in C{items} will be sent as literals if they contain CR or LF,\n    otherwise they will be quoted.  References to None in C{items} will be\n    translated to the atom NIL.  Objects with a 'read' attribute will have\n    it called on them with no arguments and the returned string will be\n    inserted into the output as a literal.  Integers will be converted to\n    strings and inserted into the output unquoted.  Instances of\n    C{DontQuoteMe} will be converted to strings and inserted into the output\n    unquoted.\n\n    This function used to be much nicer, and only quote things that really\n    needed to be quoted (and C{DontQuoteMe} did not exist), however, many\n    broken IMAP4 clients were unable to deal with this level of sophistication,\n    forcing the current behavior to be adopted for practical reasons.\n\n    @type items: Any iterable\n\n    @rtype: L{str}\n    "
    pieces = []
    for i in items:
        if isinstance(i, str):
            i = i.encode('ascii')
        if i is None:
            pieces.extend([b' ', b'NIL'])
        elif isinstance(i, int):
            pieces.extend([b' ', networkString(str(i))])
        elif isinstance(i, DontQuoteMe):
            pieces.extend([b' ', i.value])
        elif isinstance(i, bytes):
            if _needsLiteral(i):
                pieces.extend([b' ', b'{%d}' % (len(i),), IMAP4Server.delimiter, i])
            else:
                pieces.extend([b' ', _quote(i)])
        elif hasattr(i, 'read'):
            d = i.read()
            pieces.extend([b' ', b'{%d}' % (len(d),), IMAP4Server.delimiter, d])
        else:
            pieces.extend([b' ', b'(' + collapseNestedLists(i) + b')'])
    return b''.join(pieces[1:])

@implementer(IAccount)
class MemoryAccountWithoutNamespaces:
    mailboxes = None
    subscriptions = None
    top_id = 0

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.mailboxes = {}
        self.subscriptions = []

    def allocateID(self):
        if False:
            return 10
        id = self.top_id
        self.top_id += 1
        return id

    def addMailbox(self, name, mbox=None):
        if False:
            i = 10
            return i + 15
        name = _parseMbox(name.upper())
        if name in self.mailboxes:
            raise MailboxCollision(name)
        if mbox is None:
            mbox = self._emptyMailbox(name, self.allocateID())
        self.mailboxes[name] = mbox
        return 1

    def create(self, pathspec):
        if False:
            return 10
        paths = [path for path in pathspec.split('/') if path]
        for accum in range(1, len(paths)):
            try:
                self.addMailbox('/'.join(paths[:accum]))
            except MailboxCollision:
                pass
        try:
            self.addMailbox('/'.join(paths))
        except MailboxCollision:
            if not pathspec.endswith('/'):
                return False
        return True

    def _emptyMailbox(self, name, id):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def select(self, name, readwrite=1):
        if False:
            print('Hello World!')
        return self.mailboxes.get(_parseMbox(name.upper()))

    def delete(self, name):
        if False:
            for i in range(10):
                print('nop')
        name = _parseMbox(name.upper())
        mbox = self.mailboxes.get(name)
        if not mbox:
            raise MailboxException('No such mailbox')
        if '\\Noselect' in mbox.getFlags():
            for others in self.mailboxes.keys():
                if others != name and others.startswith(name):
                    raise MailboxException('Hierarchically inferior mailboxes exist and \\Noselect is set')
        mbox.destroy()
        if len(self._inferiorNames(name)) > 1:
            raise MailboxException(f'Name "{name}" has inferior hierarchical names')
        del self.mailboxes[name]

    def rename(self, oldname, newname):
        if False:
            print('Hello World!')
        oldname = _parseMbox(oldname.upper())
        newname = _parseMbox(newname.upper())
        if oldname not in self.mailboxes:
            raise NoSuchMailbox(oldname)
        inferiors = self._inferiorNames(oldname)
        inferiors = [(o, o.replace(oldname, newname, 1)) for o in inferiors]
        for (old, new) in inferiors:
            if new in self.mailboxes:
                raise MailboxCollision(new)
        for (old, new) in inferiors:
            self.mailboxes[new] = self.mailboxes[old]
            del self.mailboxes[old]

    def _inferiorNames(self, name):
        if False:
            for i in range(10):
                print('nop')
        inferiors = []
        for infname in self.mailboxes.keys():
            if infname.startswith(name):
                inferiors.append(infname)
        return inferiors

    def isSubscribed(self, name):
        if False:
            print('Hello World!')
        return _parseMbox(name.upper()) in self.subscriptions

    def subscribe(self, name):
        if False:
            i = 10
            return i + 15
        name = _parseMbox(name.upper())
        if name not in self.subscriptions:
            self.subscriptions.append(name)

    def unsubscribe(self, name):
        if False:
            print('Hello World!')
        name = _parseMbox(name.upper())
        if name not in self.subscriptions:
            raise MailboxException(f'Not currently subscribed to {name}')
        self.subscriptions.remove(name)

    def listMailboxes(self, ref, wildcard):
        if False:
            print('Hello World!')
        ref = self._inferiorNames(_parseMbox(ref.upper()))
        wildcard = wildcardToRegexp(wildcard, '/')
        return [(i, self.mailboxes[i]) for i in ref if wildcard.match(i)]

@implementer(INamespacePresenter)
class MemoryAccount(MemoryAccountWithoutNamespaces):

    def getPersonalNamespaces(self):
        if False:
            return 10
        return [[b'', b'/']]

    def getSharedNamespaces(self):
        if False:
            return 10
        return None

    def getOtherNamespaces(self):
        if False:
            while True:
                i = 10
        return None

    def getUserNamespaces(self):
        if False:
            return 10
        return None
_statusRequestDict = {'MESSAGES': 'getMessageCount', 'RECENT': 'getRecentCount', 'UIDNEXT': 'getUIDNext', 'UIDVALIDITY': 'getUIDValidity', 'UNSEEN': 'getUnseenCount'}

def statusRequestHelper(mbox, names):
    if False:
        return 10
    r = {}
    for n in names:
        r[n] = getattr(mbox, _statusRequestDict[n.upper()])()
    return r

def parseAddr(addr):
    if False:
        while True:
            i = 10
    if addr is None:
        return [(None, None, None)]
    addr = email.utils.getaddresses([addr])
    return [[fn or None, None] + address.split('@') for (fn, address) in addr]

def getEnvelope(msg):
    if False:
        return 10
    headers = msg.getHeaders(True)
    date = headers.get('date')
    subject = headers.get('subject')
    from_ = headers.get('from')
    sender = headers.get('sender', from_)
    reply_to = headers.get('reply-to', from_)
    to = headers.get('to')
    cc = headers.get('cc')
    bcc = headers.get('bcc')
    in_reply_to = headers.get('in-reply-to')
    mid = headers.get('message-id')
    return (date, subject, parseAddr(from_), parseAddr(sender), reply_to and parseAddr(reply_to), to and parseAddr(to), cc and parseAddr(cc), bcc and parseAddr(bcc), in_reply_to, mid)

def getLineCount(msg):
    if False:
        return 10
    lines = 0
    for _ in msg.getBodyFile():
        lines += 1
    return lines

def unquote(s):
    if False:
        return 10
    if s[0] == s[-1] == '"':
        return s[1:-1]
    return s

def _getContentType(msg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a two-tuple of the main and subtype of the given message.\n    '
    attrs = None
    mm = msg.getHeaders(False, 'content-type').get('content-type', '')
    mm = ''.join(mm.splitlines())
    if mm:
        mimetype = mm.split(';')
        type = mimetype[0].split('/', 1)
        if len(type) == 1:
            major = type[0]
            minor = None
        else:
            (major, minor) = type
        attrs = dict((x.strip().lower().split('=', 1) for x in mimetype[1:]))
    else:
        major = minor = None
    return (major, minor, attrs)

def _getMessageStructure(message):
    if False:
        while True:
            i = 10
    '\n    Construct an appropriate type of message structure object for the given\n    message object.\n\n    @param message: A L{IMessagePart} provider\n\n    @return: A L{_MessageStructure} instance of the most specific type available\n        for the given message, determined by inspecting the MIME type of the\n        message.\n    '
    (main, subtype, attrs) = _getContentType(message)
    if main is not None:
        main = main.lower()
    if subtype is not None:
        subtype = subtype.lower()
    if main == 'multipart':
        return _MultipartMessageStructure(message, subtype, attrs)
    elif (main, subtype) == ('message', 'rfc822'):
        return _RFC822MessageStructure(message, main, subtype, attrs)
    elif main == 'text':
        return _TextMessageStructure(message, main, subtype, attrs)
    else:
        return _SinglepartMessageStructure(message, main, subtype, attrs)

class _MessageStructure:
    """
    L{_MessageStructure} is a helper base class for message structure classes
    representing the structure of particular kinds of messages, as defined by
    their MIME type.
    """

    def __init__(self, message, attrs):
        if False:
            i = 10
            return i + 15
        '\n        @param message: An L{IMessagePart} provider which this structure object\n            reports on.\n\n        @param attrs: A C{dict} giving the parameters of the I{Content-Type}\n            header of the message.\n        '
        self.message = message
        self.attrs = attrs

    def _disposition(self, disp):
        if False:
            while True:
                i = 10
        '\n        Parse a I{Content-Disposition} header into a two-sequence of the\n        disposition and a flattened list of its parameters.\n\n        @return: L{None} if there is no disposition header value, a L{list} with\n            two elements otherwise.\n        '
        if disp:
            disp = disp.split('; ')
            if len(disp) == 1:
                disp = (disp[0].lower(), None)
            elif len(disp) > 1:
                params = [x for param in disp[1:] for x in param.split('=', 1)]
                disp = [disp[0].lower(), params]
            return disp
        else:
            return None

    def _unquotedAttrs(self):
        if False:
            print('Hello World!')
        '\n        @return: The I{Content-Type} parameters, unquoted, as a flat list with\n            each Nth element giving a parameter name and N+1th element giving\n            the corresponding parameter value.\n        '
        if self.attrs:
            unquoted = [(k, unquote(v)) for (k, v) in self.attrs.items()]
            return [y for x in sorted(unquoted) for y in x]
        return None

class _SinglepartMessageStructure(_MessageStructure):
    """
    L{_SinglepartMessageStructure} represents the message structure of a
    non-I{multipart/*} message.
    """
    _HEADERS = ['content-id', 'content-description', 'content-transfer-encoding']

    def __init__(self, message, main, subtype, attrs):
        if False:
            return 10
        '\n        @param message: An L{IMessagePart} provider which this structure object\n            reports on.\n\n        @param main: A L{str} giving the main MIME type of the message (for\n            example, C{"text"}).\n\n        @param subtype: A L{str} giving the MIME subtype of the message (for\n            example, C{"plain"}).\n\n        @param attrs: A C{dict} giving the parameters of the I{Content-Type}\n            header of the message.\n        '
        _MessageStructure.__init__(self, message, attrs)
        self.main = main
        self.subtype = subtype
        self.attrs = attrs

    def _basicFields(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a list of the basic fields for a single-part message.\n        '
        headers = self.message.getHeaders(False, *self._HEADERS)
        size = self.message.getSize()
        (major, minor) = (self.main, self.subtype)
        unquotedAttrs = self._unquotedAttrs()
        return [major, minor, unquotedAttrs, headers.get('content-id'), headers.get('content-description'), headers.get('content-transfer-encoding'), size]

    def encode(self, extended):
        if False:
            while True:
                i = 10
        '\n        Construct and return a list of the basic and extended fields for a\n        single-part message.  The list suitable to be encoded into a BODY or\n        BODYSTRUCTURE response.\n        '
        result = self._basicFields()
        if extended:
            result.extend(self._extended())
        return result

    def _extended(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The extension data of a non-multipart body part are in the\n        following order:\n\n          1. body MD5\n\n             A string giving the body MD5 value as defined in [MD5].\n\n          2. body disposition\n\n             A parenthesized list with the same content and function as\n             the body disposition for a multipart body part.\n\n          3. body language\n\n             A string or parenthesized list giving the body language\n             value as defined in [LANGUAGE-TAGS].\n\n          4. body location\n\n             A string list giving the body content URI as defined in\n             [LOCATION].\n\n        '
        result = []
        headers = self.message.getHeaders(False, 'content-md5', 'content-disposition', 'content-language', 'content-language')
        result.append(headers.get('content-md5'))
        result.append(self._disposition(headers.get('content-disposition')))
        result.append(headers.get('content-language'))
        result.append(headers.get('content-location'))
        return result

class _TextMessageStructure(_SinglepartMessageStructure):
    """
    L{_TextMessageStructure} represents the message structure of a I{text/*}
    message.
    """

    def encode(self, extended):
        if False:
            print('Hello World!')
        '\n        A body type of type TEXT contains, immediately after the basic\n        fields, the size of the body in text lines.  Note that this\n        size is the size in its content transfer encoding and not the\n        resulting size after any decoding.\n        '
        result = _SinglepartMessageStructure._basicFields(self)
        result.append(getLineCount(self.message))
        if extended:
            result.extend(self._extended())
        return result

class _RFC822MessageStructure(_SinglepartMessageStructure):
    """
    L{_RFC822MessageStructure} represents the message structure of a
    I{message/rfc822} message.
    """

    def encode(self, extended):
        if False:
            i = 10
            return i + 15
        '\n        A body type of type MESSAGE and subtype RFC822 contains,\n        immediately after the basic fields, the envelope structure,\n        body structure, and size in text lines of the encapsulated\n        message.\n        '
        result = _SinglepartMessageStructure.encode(self, extended)
        contained = self.message.getSubPart(0)
        result.append(getEnvelope(contained))
        result.append(getBodyStructure(contained, False))
        result.append(getLineCount(contained))
        return result

class _MultipartMessageStructure(_MessageStructure):
    """
    L{_MultipartMessageStructure} represents the message structure of a
    I{multipart/*} message.
    """

    def __init__(self, message, subtype, attrs):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param message: An L{IMessagePart} provider which this structure object\n            reports on.\n\n        @param subtype: A L{str} giving the MIME subtype of the message (for\n            example, C{"plain"}).\n\n        @param attrs: A C{dict} giving the parameters of the I{Content-Type}\n            header of the message.\n        '
        _MessageStructure.__init__(self, message, attrs)
        self.subtype = subtype

    def _getParts(self):
        if False:
            return 10
        '\n        Return an iterator over all of the sub-messages of this message.\n        '
        i = 0
        while True:
            try:
                part = self.message.getSubPart(i)
            except IndexError:
                break
            else:
                yield part
                i += 1

    def encode(self, extended):
        if False:
            for i in range(10):
                print('nop')
        '\n        Encode each sub-message and added the additional I{multipart} fields.\n        '
        result = [_getMessageStructure(p).encode(extended) for p in self._getParts()]
        result.append(self.subtype)
        if extended:
            result.extend(self._extended())
        return result

    def _extended(self):
        if False:
            i = 10
            return i + 15
        '\n        The extension data of a multipart body part are in the following order:\n\n          1. body parameter parenthesized list\n               A parenthesized list of attribute/value pairs [e.g., ("foo"\n               "bar" "baz" "rag") where "bar" is the value of "foo", and\n               "rag" is the value of "baz"] as defined in [MIME-IMB].\n\n          2. body disposition\n               A parenthesized list, consisting of a disposition type\n               string, followed by a parenthesized list of disposition\n               attribute/value pairs as defined in [DISPOSITION].\n\n          3. body language\n               A string or parenthesized list giving the body language\n               value as defined in [LANGUAGE-TAGS].\n\n          4. body location\n               A string list giving the body content URI as defined in\n               [LOCATION].\n        '
        result = []
        headers = self.message.getHeaders(False, 'content-language', 'content-location', 'content-disposition')
        result.append(self._unquotedAttrs())
        result.append(self._disposition(headers.get('content-disposition')))
        result.append(headers.get('content-language', None))
        result.append(headers.get('content-location', None))
        return result

def getBodyStructure(msg, extended=False):
    if False:
        i = 10
        return i + 15
    '\n    RFC 3501, 7.4.2, BODYSTRUCTURE::\n\n      A parenthesized list that describes the [MIME-IMB] body structure of a\n      message.  This is computed by the server by parsing the [MIME-IMB] header\n      fields, defaulting various fields as necessary.\n\n        For example, a simple text message of 48 lines and 2279 octets can have\n        a body structure of: ("TEXT" "PLAIN" ("CHARSET" "US-ASCII") NIL NIL\n        "7BIT" 2279 48)\n\n    This is represented as::\n\n        ["TEXT", "PLAIN", ["CHARSET", "US-ASCII"], None, None, "7BIT", 2279, 48]\n\n    These basic fields are documented in the RFC as:\n\n      1. body type\n\n         A string giving the content media type name as defined in\n         [MIME-IMB].\n\n      2. body subtype\n\n         A string giving the content subtype name as defined in\n         [MIME-IMB].\n\n      3. body parameter parenthesized list\n\n         A parenthesized list of attribute/value pairs [e.g., ("foo"\n         "bar" "baz" "rag") where "bar" is the value of "foo" and\n         "rag" is the value of "baz"] as defined in [MIME-IMB].\n\n      4. body id\n\n         A string giving the content id as defined in [MIME-IMB].\n\n      5. body description\n\n         A string giving the content description as defined in\n         [MIME-IMB].\n\n      6. body encoding\n\n         A string giving the content transfer encoding as defined in\n         [MIME-IMB].\n\n      7. body size\n\n         A number giving the size of the body in octets.  Note that this size is\n         the size in its transfer encoding and not the resulting size after any\n         decoding.\n\n    Put another way, the body structure is a list of seven elements.  The\n    semantics of the elements of this list are:\n\n       1. Byte string giving the major MIME type\n       2. Byte string giving the minor MIME type\n       3. A list giving the Content-Type parameters of the message\n       4. A byte string giving the content identifier for the message part, or\n          None if it has no content identifier.\n       5. A byte string giving the content description for the message part, or\n          None if it has no content description.\n       6. A byte string giving the Content-Encoding of the message body\n       7. An integer giving the number of octets in the message body\n\n    The RFC goes on::\n\n        Multiple parts are indicated by parenthesis nesting.  Instead of a body\n        type as the first element of the parenthesized list, there is a sequence\n        of one or more nested body structures.  The second element of the\n        parenthesized list is the multipart subtype (mixed, digest, parallel,\n        alternative, etc.).\n\n        For example, a two part message consisting of a text and a\n        BASE64-encoded text attachment can have a body structure of: (("TEXT"\n        "PLAIN" ("CHARSET" "US-ASCII") NIL NIL "7BIT" 1152 23)("TEXT" "PLAIN"\n        ("CHARSET" "US-ASCII" "NAME" "cc.diff")\n        "<960723163407.20117h@cac.washington.edu>" "Compiler diff" "BASE64" 4554\n        73) "MIXED")\n\n    This is represented as::\n\n        [["TEXT", "PLAIN", ["CHARSET", "US-ASCII"], None, None, "7BIT", 1152,\n          23],\n         ["TEXT", "PLAIN", ["CHARSET", "US-ASCII", "NAME", "cc.diff"],\n          "<960723163407.20117h@cac.washington.edu>", "Compiler diff",\n          "BASE64", 4554, 73],\n         "MIXED"]\n\n    In other words, a list of N + 1 elements, where N is the number of parts in\n    the message.  The first N elements are structures as defined by the previous\n    section.  The last element is the minor MIME subtype of the multipart\n    message.\n\n    Additionally, the RFC describes extension data::\n\n        Extension data follows the multipart subtype.  Extension data is never\n        returned with the BODY fetch, but can be returned with a BODYSTRUCTURE\n        fetch.  Extension data, if present, MUST be in the defined order.\n\n    The C{extended} flag controls whether extension data might be returned with\n    the normal data.\n    '
    return _getMessageStructure(msg).encode(extended)

def _formatHeaders(headers):
    if False:
        for i in range(10):
            print('nop')
    hdrs = [': '.join((k.title(), '\r\n'.join(v.splitlines()))) for (k, v) in headers.items()]
    hdrs = '\r\n'.join(hdrs) + '\r\n'
    return networkString(hdrs)

def subparts(m):
    if False:
        print('Hello World!')
    i = 0
    try:
        while True:
            yield m.getSubPart(i)
            i += 1
    except IndexError:
        pass

def iterateInReactor(i):
    if False:
        i = 10
        return i + 15
    '\n    Consume an interator at most a single iteration per reactor iteration.\n\n    If the iterator produces a Deferred, the next iteration will not occur\n    until the Deferred fires, otherwise the next iteration will be taken\n    in the next reactor iteration.\n\n    @rtype: C{Deferred}\n    @return: A deferred which fires (with None) when the iterator is\n    exhausted or whose errback is called if there is an exception.\n    '
    from twisted.internet import reactor
    d = defer.Deferred()

    def go(last):
        if False:
            i = 10
            return i + 15
        try:
            r = next(i)
        except StopIteration:
            d.callback(last)
        except BaseException:
            d.errback()
        else:
            if isinstance(r, defer.Deferred):
                r.addCallback(go)
            else:
                reactor.callLater(0, go, r)
    go(None)
    return d

class MessageProducer:
    CHUNK_SIZE = 2 ** 2 ** 2 ** 2
    _uuid4 = staticmethod(uuid.uuid4)

    def __init__(self, msg, buffer=None, scheduler=None):
        if False:
            print('Hello World!')
        '\n        Produce this message.\n\n        @param msg: The message I am to produce.\n        @type msg: L{IMessage}\n\n        @param buffer: A buffer to hold the message in.  If None, I will\n            use a L{tempfile.TemporaryFile}.\n        @type buffer: file-like\n        '
        self.msg = msg
        if buffer is None:
            buffer = tempfile.TemporaryFile()
        self.buffer = buffer
        if scheduler is None:
            scheduler = iterateInReactor
        self.scheduler = scheduler
        self.write = self.buffer.write

    def beginProducing(self, consumer):
        if False:
            return 10
        self.consumer = consumer
        return self.scheduler(self._produce())

    def _produce(self):
        if False:
            while True:
                i = 10
        headers = self.msg.getHeaders(True)
        boundary = None
        if self.msg.isMultipart():
            content = headers.get('content-type')
            parts = [x.split('=', 1) for x in content.split(';')[1:]]
            parts = {k.lower().strip(): v for (k, v) in parts}
            boundary = parts.get('boundary')
            if boundary is None:
                boundary = f'----={self._uuid4().hex}'
                headers['content-type'] += f'; boundary="{boundary}"'
            elif boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            boundary = networkString(boundary)
        self.write(_formatHeaders(headers))
        self.write(b'\r\n')
        if self.msg.isMultipart():
            for p in subparts(self.msg):
                self.write(b'\r\n--' + boundary + b'\r\n')
                yield MessageProducer(p, self.buffer, self.scheduler).beginProducing(None)
            self.write(b'\r\n--' + boundary + b'--\r\n')
        else:
            f = self.msg.getBodyFile()
            while True:
                b = f.read(self.CHUNK_SIZE)
                if b:
                    self.buffer.write(b)
                    yield None
                else:
                    break
        if self.consumer:
            self.buffer.seek(0, 0)
            yield FileProducer(self.buffer).beginProducing(self.consumer).addCallback(lambda _: self)

class _FetchParser:

    class Envelope:
        type = 'envelope'
        __str__ = lambda self: 'envelope'

    class Flags:
        type = 'flags'
        __str__ = lambda self: 'flags'

    class InternalDate:
        type = 'internaldate'
        __str__ = lambda self: 'internaldate'

    class RFC822Header:
        type = 'rfc822header'
        __str__ = lambda self: 'rfc822.header'

    class RFC822Text:
        type = 'rfc822text'
        __str__ = lambda self: 'rfc822.text'

    class RFC822Size:
        type = 'rfc822size'
        __str__ = lambda self: 'rfc822.size'

    class RFC822:
        type = 'rfc822'
        __str__ = lambda self: 'rfc822'

    class UID:
        type = 'uid'
        __str__ = lambda self: 'uid'

    class Body:
        type = 'body'
        peek = False
        header = None
        mime = None
        text = None
        part = ()
        empty = False
        partialBegin = None
        partialLength = None

        def __str__(self) -> str:
            if False:
                return 10
            return self.__bytes__().decode('ascii')

        def __bytes__(self) -> bytes:
            if False:
                return 10
            base = b'BODY'
            part = b''
            separator = b''
            if self.part:
                part = b'.'.join([str(x + 1).encode('ascii') for x in self.part])
                separator = b'.'
            if self.header:
                base += b'[' + part + separator + str(self.header).encode('ascii') + b']'
            elif self.text:
                base += b'[' + part + separator + b'TEXT]'
            elif self.mime:
                base += b'[' + part + separator + b'MIME]'
            elif self.empty:
                base += b'[' + part + b']'
            if self.partialBegin is not None:
                base += b'<%d.%d>' % (self.partialBegin, self.partialLength)
            return base

    class BodyStructure:
        type = 'bodystructure'
        __str__ = lambda self: 'bodystructure'

    class Header:
        negate = False
        fields = None
        part = None

        def __str__(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return self.__bytes__().decode('ascii')

        def __bytes__(self) -> bytes:
            if False:
                return 10
            base = b'HEADER'
            if self.fields:
                base += b'.FIELDS'
                if self.negate:
                    base += b'.NOT'
                fields = []
                for f in self.fields:
                    f = f.title()
                    if _needsQuote(f):
                        f = _quote(f)
                    fields.append(f)
                base += b' (' + b' '.join(fields) + b')'
            if self.part:
                base = b'.'.join([(x + 1).__bytes__() for x in self.part]) + b'.' + base
            return base

    class Text:
        pass

    class MIME:
        pass
    parts = None
    _simple_fetch_att = [(b'envelope', Envelope), (b'flags', Flags), (b'internaldate', InternalDate), (b'rfc822.header', RFC822Header), (b'rfc822.text', RFC822Text), (b'rfc822.size', RFC822Size), (b'rfc822', RFC822), (b'uid', UID), (b'bodystructure', BodyStructure)]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = ['initial']
        self.result = []
        self.remaining = b''

    def parseString(self, s):
        if False:
            while True:
                i = 10
        s = self.remaining + s
        try:
            while s or self.state:
                if not self.state:
                    raise IllegalClientResponse('Invalid Argument')
                state = self.state.pop()
                try:
                    used = getattr(self, 'state_' + state)(s)
                except BaseException:
                    self.state.append(state)
                    raise
                else:
                    s = s[used:]
        finally:
            self.remaining = s

    def state_initial(self, s):
        if False:
            print('Hello World!')
        if s == b'':
            return 0
        l = s.lower()
        if l.startswith(b'all'):
            self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size(), self.Envelope()))
            return 3
        if l.startswith(b'full'):
            self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size(), self.Envelope(), self.Body()))
            return 4
        if l.startswith(b'fast'):
            self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size()))
            return 4
        if l.startswith(b'('):
            self.state.extend(('close_paren', 'maybe_fetch_att', 'fetch_att'))
            return 1
        self.state.append('fetch_att')
        return 0

    def state_close_paren(self, s):
        if False:
            return 10
        if s.startswith(b')'):
            return 1
        raise Exception('Missing )')

    def state_whitespace(self, s):
        if False:
            print('Hello World!')
        if not s or not s[0:1].isspace():
            raise Exception('Whitespace expected, none found')
        i = 0
        for i in range(len(s)):
            if not s[i:i + 1].isspace():
                break
        return i

    def state_maybe_fetch_att(self, s):
        if False:
            i = 10
            return i + 15
        if not s.startswith(b')'):
            self.state.extend(('maybe_fetch_att', 'fetch_att', 'whitespace'))
        return 0

    def state_fetch_att(self, s):
        if False:
            while True:
                i = 10
        l = s.lower()
        for (name, cls) in self._simple_fetch_att:
            if l.startswith(name):
                self.result.append(cls())
                return len(name)
        b = self.Body()
        if l.startswith(b'body.peek'):
            b.peek = True
            used = 9
        elif l.startswith(b'body'):
            used = 4
        else:
            raise Exception(f'Nothing recognized in fetch_att: {l}')
        self.pending_body = b
        self.state.extend(('got_body', 'maybe_partial', 'maybe_section'))
        return used

    def state_got_body(self, s):
        if False:
            i = 10
            return i + 15
        self.result.append(self.pending_body)
        del self.pending_body
        return 0

    def state_maybe_section(self, s):
        if False:
            for i in range(10):
                print('nop')
        if not s.startswith(b'['):
            return 0
        self.state.extend(('section', 'part_number'))
        return 1
    _partExpr = re.compile(b'(\\d+(?:\\.\\d+)*)\\.?')

    def state_part_number(self, s):
        if False:
            return 10
        m = self._partExpr.match(s)
        if m is not None:
            self.parts = [int(p) - 1 for p in m.groups()[0].split(b'.')]
            return m.end()
        else:
            self.parts = []
            return 0

    def state_section(self, s):
        if False:
            while True:
                i = 10
        l = s.lower()
        used = 0
        if l.startswith(b']'):
            self.pending_body.empty = True
            used += 1
        elif l.startswith(b'header]'):
            h = self.pending_body.header = self.Header()
            h.negate = True
            h.fields = ()
            used += 7
        elif l.startswith(b'text]'):
            self.pending_body.text = self.Text()
            used += 5
        elif l.startswith(b'mime]'):
            self.pending_body.mime = self.MIME()
            used += 5
        else:
            h = self.Header()
            if l.startswith(b'header.fields.not'):
                h.negate = True
                used += 17
            elif l.startswith(b'header.fields'):
                used += 13
            else:
                raise Exception(f'Unhandled section contents: {l!r}')
            self.pending_body.header = h
            self.state.extend(('finish_section', 'header_list', 'whitespace'))
        self.pending_body.part = tuple(self.parts)
        self.parts = None
        return used

    def state_finish_section(self, s):
        if False:
            print('Hello World!')
        if not s.startswith(b']'):
            raise Exception('section must end with ]')
        return 1

    def state_header_list(self, s):
        if False:
            for i in range(10):
                print('nop')
        if not s.startswith(b'('):
            raise Exception('Header list must begin with (')
        end = s.find(b')')
        if end == -1:
            raise Exception('Header list must end with )')
        headers = s[1:end].split()
        self.pending_body.header.fields = [h.upper() for h in headers]
        return end + 1

    def state_maybe_partial(self, s):
        if False:
            for i in range(10):
                print('nop')
        if not s.startswith(b'<'):
            return 0
        end = s.find(b'>')
        if end == -1:
            raise Exception('Found < but not >')
        partial = s[1:end]
        parts = partial.split(b'.', 1)
        if len(parts) != 2:
            raise Exception('Partial specification did not include two .-delimited integers')
        (begin, length) = map(int, parts)
        self.pending_body.partialBegin = begin
        self.pending_body.partialLength = length
        return end + 1

class FileProducer:
    CHUNK_SIZE = 2 ** 2 ** 2 ** 2
    firstWrite = True

    def __init__(self, f):
        if False:
            for i in range(10):
                print('nop')
        self.f = f

    def beginProducing(self, consumer):
        if False:
            return 10
        self.consumer = consumer
        self.produce = consumer.write
        d = self._onDone = defer.Deferred()
        self.consumer.registerProducer(self, False)
        return d

    def resumeProducing(self):
        if False:
            while True:
                i = 10
        b = b''
        if self.firstWrite:
            b = b'{%d}\r\n' % (self._size(),)
            self.firstWrite = False
        if not self.f:
            return
        b = b + self.f.read(self.CHUNK_SIZE)
        if not b:
            self.consumer.unregisterProducer()
            self._onDone.callback(self)
            self._onDone = self.f = self.consumer = None
        else:
            self.produce(b)

    def pauseProducing(self):
        if False:
            print('Hello World!')
        '\n        Pause the producer.  This does nothing.\n        '

    def stopProducing(self):
        if False:
            return 10
        '\n        Stop the producer.  This does nothing.\n        '

    def _size(self):
        if False:
            return 10
        b = self.f.tell()
        self.f.seek(0, 2)
        e = self.f.tell()
        self.f.seek(b, 0)
        return e - b

def parseTime(s):
    if False:
        i = 10
        return i + 15
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    expr = {'day': '(?P<day>3[0-1]|[1-2]\\d|0[1-9]|[1-9]| [1-9])', 'mon': '(?P<mon>\\w+)', 'year': '(?P<year>\\d\\d\\d\\d)'}
    m = re.match('%(day)s-%(mon)s-%(year)s' % expr, s)
    if not m:
        raise ValueError(f'Cannot parse time string {s!r}')
    d = m.groupdict()
    try:
        d['mon'] = 1 + months.index(d['mon'].lower()) % 12
        d['year'] = int(d['year'])
        d['day'] = int(d['day'])
    except ValueError:
        raise ValueError(f'Cannot parse time string {s!r}')
    else:
        return time.struct_time((d['year'], d['mon'], d['day'], 0, 0, 0, -1, -1, -1))
memory_cast = getattr(memoryview, 'cast', lambda *x: x[0])

def modified_base64(s):
    if False:
        i = 10
        return i + 15
    s_utf7 = s.encode('utf-7')
    return s_utf7[1:-1].replace(b'/', b',')

def modified_unbase64(s):
    if False:
        for i in range(10):
            print('nop')
    s_utf7 = b'+' + s.replace(b',', b'/') + b'-'
    return s_utf7.decode('utf-7')

def encoder(s, errors=None):
    if False:
        while True:
            i = 10
    '\n    Encode the given C{unicode} string using the IMAP4 specific variation of\n    UTF-7.\n\n    @type s: C{unicode}\n    @param s: The text to encode.\n\n    @param errors: Policy for handling encoding errors.  Currently ignored.\n\n    @return: L{tuple} of a L{str} giving the encoded bytes and an L{int}\n        giving the number of code units consumed from the input.\n    '
    r = bytearray()
    _in = []
    valid_chars = set(map(chr, range(32, 127))) - {'&'}
    for c in s:
        if c in valid_chars:
            if _in:
                r += b'&' + modified_base64(''.join(_in)) + b'-'
                del _in[:]
            r.append(ord(c))
        elif c == '&':
            if _in:
                r += b'&' + modified_base64(''.join(_in)) + b'-'
                del _in[:]
            r += b'&-'
        else:
            _in.append(c)
    if _in:
        r.extend(b'&' + modified_base64(''.join(_in)) + b'-')
    return (bytes(r), len(s))

def decoder(s, errors=None):
    if False:
        i = 10
        return i + 15
    '\n    Decode the given L{str} using the IMAP4 specific variation of UTF-7.\n\n    @type s: L{str}\n    @param s: The bytes to decode.\n\n    @param errors: Policy for handling decoding errors.  Currently ignored.\n\n    @return: a L{tuple} of a C{unicode} string giving the text which was\n        decoded and an L{int} giving the number of bytes consumed from the\n        input.\n    '
    r = []
    decode = []
    s = memory_cast(memoryview(s), 'c')
    for c in s:
        if c == b'&' and (not decode):
            decode.append(b'&')
        elif c == b'-' and decode:
            if len(decode) == 1:
                r.append('&')
            else:
                r.append(modified_unbase64(b''.join(decode[1:])))
            decode = []
        elif decode:
            decode.append(c)
        else:
            r.append(c.decode())
    if decode:
        r.append(modified_unbase64(b''.join(decode[1:])))
    return (''.join(r), len(s))

class StreamReader(codecs.StreamReader):

    def decode(self, s, errors='strict'):
        if False:
            while True:
                i = 10
        return decoder(s)

class StreamWriter(codecs.StreamWriter):

    def encode(self, s, errors='strict'):
        if False:
            return 10
        return encoder(s)
_codecInfo = codecs.CodecInfo(encoder, decoder, StreamReader, StreamWriter)

def imap4_utf_7(name):
    if False:
        i = 10
        return i + 15
    if name.replace('-', '_') == 'imap4_utf_7':
        return _codecInfo
codecs.register(imap4_utf_7)
__all__ = ['IMAP4Server', 'IMAP4Client', 'IMailboxListener', 'IClientAuthentication', 'IAccount', 'IMailbox', 'INamespacePresenter', 'ICloseableMailbox', 'IMailboxInfo', 'IMessage', 'IMessageCopier', 'IMessageFile', 'ISearchableMailbox', 'IMessagePart', 'IMAP4Exception', 'IllegalClientResponse', 'IllegalOperation', 'IllegalMailboxEncoding', 'UnhandledResponse', 'NegativeResponse', 'NoSupportedAuthentication', 'IllegalServerResponse', 'IllegalIdentifierError', 'IllegalQueryError', 'MismatchedNesting', 'MismatchedQuoting', 'MailboxException', 'MailboxCollision', 'NoSuchMailbox', 'ReadOnlyMailbox', 'CramMD5ClientAuthenticator', 'PLAINAuthenticator', 'LOGINAuthenticator', 'PLAINCredentials', 'LOGINCredentials', 'Query', 'Not', 'Or', 'MemoryAccount', 'statusRequestHelper']