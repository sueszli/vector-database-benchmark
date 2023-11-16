"""
Banana -- s-exp based protocol.

Future Plans: This module is almost entirely stable.  The same caveat applies
to it as applies to L{twisted.spread.jelly}, however.  Read its future plans
for more details.

@author: Glyph Lefkowitz
"""
import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName

class BananaError(Exception):
    pass

def int2b128(integer, stream):
    if False:
        return 10
    if integer == 0:
        stream(b'\x00')
        return
    assert integer > 0, 'can only encode positive integers'
    while integer:
        stream(bytes((integer & 127,)))
        integer = integer >> 7

def b1282int(st):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an integer represented as a base 128 string into an L{int}.\n\n    @param st: The integer encoded in a byte string.\n    @type st: L{bytes}\n\n    @return: The integer value extracted from the byte string.\n    @rtype: L{int}\n    '
    e = 1
    i = 0
    for char in iterbytes(st):
        n = ord(char)
        i += n * e
        e <<= 7
    return i
LIST = b'\x80'
INT = b'\x81'
STRING = b'\x82'
NEG = b'\x83'
FLOAT = b'\x84'
LONGINT = b'\x85'
LONGNEG = b'\x86'
VOCAB = b'\x87'
HIGH_BIT_SET = b'\x80'

def setPrefixLimit(limit):
    if False:
        while True:
            i = 10
    '\n    Set the limit on the prefix length for all Banana connections\n    established after this call.\n\n    The prefix length limit determines how many bytes of prefix a banana\n    decoder will allow before rejecting a potential object as too large.\n\n    @type limit: L{int}\n    @param limit: The number of bytes of prefix for banana to allow when\n    decoding.\n    '
    global _PREFIX_LIMIT
    _PREFIX_LIMIT = limit
_PREFIX_LIMIT = None
setPrefixLimit(64)
SIZE_LIMIT = 640 * 1024

class Banana(protocol.Protocol, styles.Ephemeral):
    """
    L{Banana} implements the I{Banana} s-expression protocol, client and
    server.

    @ivar knownDialects: These are the profiles supported by this Banana
        implementation.
    @type knownDialects: L{list} of L{bytes}
    """
    knownDialects = [b'pb', b'none']
    prefixLimit = None
    sizeLimit = SIZE_LIMIT

    def setPrefixLimit(self, limit):
        if False:
            return 10
        '\n        Set the prefix limit for decoding done by this protocol instance.\n\n        @see: L{setPrefixLimit}\n        '
        self.prefixLimit = limit
        self._smallestLongInt = -2 ** (limit * 7) + 1
        self._smallestInt = -2 ** 31
        self._largestInt = 2 ** 31 - 1
        self._largestLongInt = 2 ** (limit * 7) - 1

    def connectionReady(self):
        if False:
            print('Hello World!')
        'Surrogate for connectionMade\n        Called after protocol negotiation.\n        '

    def _selectDialect(self, dialect):
        if False:
            i = 10
            return i + 15
        self.currentDialect = dialect
        self.connectionReady()

    def callExpressionReceived(self, obj):
        if False:
            return 10
        if self.currentDialect:
            self.expressionReceived(obj)
        elif self.isClient:
            for serverVer in obj:
                if serverVer in self.knownDialects:
                    self.sendEncoded(serverVer)
                    self._selectDialect(serverVer)
                    break
            else:
                log.msg("The client doesn't speak any of the protocols offered by the server: disconnecting.")
                self.transport.loseConnection()
        elif obj in self.knownDialects:
            self._selectDialect(obj)
        else:
            log.msg("The client selected a protocol the server didn't suggest and doesn't know: disconnecting.")
            self.transport.loseConnection()

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.setPrefixLimit(_PREFIX_LIMIT)
        self.currentDialect = None
        if not self.isClient:
            self.sendEncoded(self.knownDialects)

    def gotItem(self, item):
        if False:
            print('Hello World!')
        l = self.listStack
        if l:
            l[-1][1].append(item)
        else:
            self.callExpressionReceived(item)
    buffer = b''

    def dataReceived(self, chunk):
        if False:
            return 10
        buffer = self.buffer + chunk
        listStack = self.listStack
        gotItem = self.gotItem
        while buffer:
            assert self.buffer != buffer, "This ain't right: {} {}".format(repr(self.buffer), repr(buffer))
            self.buffer = buffer
            pos = 0
            for ch in iterbytes(buffer):
                if ch >= HIGH_BIT_SET:
                    break
                pos = pos + 1
            else:
                if pos > self.prefixLimit:
                    raise BananaError('Security precaution: more than %d bytes of prefix' % (self.prefixLimit,))
                return
            num = buffer[:pos]
            typebyte = buffer[pos:pos + 1]
            rest = buffer[pos + 1:]
            if len(num) > self.prefixLimit:
                raise BananaError('Security precaution: longer than %d bytes worth of prefix' % (self.prefixLimit,))
            if typebyte == LIST:
                num = b1282int(num)
                if num > SIZE_LIMIT:
                    raise BananaError('Security precaution: List too long.')
                listStack.append((num, []))
                buffer = rest
            elif typebyte == STRING:
                num = b1282int(num)
                if num > SIZE_LIMIT:
                    raise BananaError('Security precaution: String too long.')
                if len(rest) >= num:
                    buffer = rest[num:]
                    gotItem(rest[:num])
                else:
                    return
            elif typebyte == INT:
                buffer = rest
                num = b1282int(num)
                gotItem(num)
            elif typebyte == LONGINT:
                buffer = rest
                num = b1282int(num)
                gotItem(num)
            elif typebyte == LONGNEG:
                buffer = rest
                num = b1282int(num)
                gotItem(-num)
            elif typebyte == NEG:
                buffer = rest
                num = -b1282int(num)
                gotItem(num)
            elif typebyte == VOCAB:
                buffer = rest
                num = b1282int(num)
                item = self.incomingVocabulary[num]
                if self.currentDialect == b'pb':
                    gotItem(item)
                else:
                    raise NotImplementedError(f'Invalid item for pb protocol {item!r}')
            elif typebyte == FLOAT:
                if len(rest) >= 8:
                    buffer = rest[8:]
                    gotItem(struct.unpack('!d', rest[:8])[0])
                else:
                    return
            else:
                raise NotImplementedError(f'Invalid Type Byte {typebyte!r}')
            while listStack and len(listStack[-1][1]) == listStack[-1][0]:
                item = listStack.pop()[1]
                gotItem(item)
        self.buffer = b''

    def expressionReceived(self, lst):
        if False:
            i = 10
            return i + 15
        'Called when an expression (list, string, or int) is received.'
        raise NotImplementedError()
    outgoingVocabulary = {b'None': 1, b'class': 2, b'dereference': 3, b'reference': 4, b'dictionary': 5, b'function': 6, b'instance': 7, b'list': 8, b'module': 9, b'persistent': 10, b'tuple': 11, b'unpersistable': 12, b'copy': 13, b'cache': 14, b'cached': 15, b'remote': 16, b'local': 17, b'lcache': 18, b'version': 19, b'login': 20, b'password': 21, b'challenge': 22, b'logged_in': 23, b'not_logged_in': 24, b'cachemessage': 25, b'message': 26, b'answer': 27, b'error': 28, b'decref': 29, b'decache': 30, b'uncache': 31}
    incomingVocabulary = {}
    for (k, v) in outgoingVocabulary.items():
        incomingVocabulary[v] = k

    def __init__(self, isClient=1):
        if False:
            print('Hello World!')
        self.listStack = []
        self.outgoingSymbols = copy.copy(self.outgoingVocabulary)
        self.outgoingSymbolCount = 0
        self.isClient = isClient

    def sendEncoded(self, obj):
        if False:
            return 10
        '\n        Send the encoded representation of the given object:\n\n        @param obj: An object to encode and send.\n\n        @raise BananaError: If the given object is not an instance of one of\n            the types supported by Banana.\n\n        @return: L{None}\n        '
        encodeStream = BytesIO()
        self._encode(obj, encodeStream.write)
        value = encodeStream.getvalue()
        self.transport.write(value)

    def _encode(self, obj, write):
        if False:
            while True:
                i = 10
        if isinstance(obj, (list, tuple)):
            if len(obj) > SIZE_LIMIT:
                raise BananaError('list/tuple is too long to send (%d)' % (len(obj),))
            int2b128(len(obj), write)
            write(LIST)
            for elem in obj:
                self._encode(elem, write)
        elif isinstance(obj, int):
            if obj < self._smallestLongInt or obj > self._largestLongInt:
                raise BananaError('int is too large to send (%d)' % (obj,))
            if obj < self._smallestInt:
                int2b128(-obj, write)
                write(LONGNEG)
            elif obj < 0:
                int2b128(-obj, write)
                write(NEG)
            elif obj <= self._largestInt:
                int2b128(obj, write)
                write(INT)
            else:
                int2b128(obj, write)
                write(LONGINT)
        elif isinstance(obj, float):
            write(FLOAT)
            write(struct.pack('!d', obj))
        elif isinstance(obj, bytes):
            if self.currentDialect == b'pb' and obj in self.outgoingSymbols:
                symbolID = self.outgoingSymbols[obj]
                int2b128(symbolID, write)
                write(VOCAB)
            else:
                if len(obj) > SIZE_LIMIT:
                    raise BananaError('byte string is too long to send (%d)' % (len(obj),))
                int2b128(len(obj), write)
                write(STRING)
                write(obj)
        else:
            raise BananaError('Banana cannot send {} objects: {!r}'.format(fullyQualifiedName(type(obj)), obj))
_i = Banana()
_i.connectionMade()
_i._selectDialect(b'none')

def encode(lst):
    if False:
        while True:
            i = 10
    'Encode a list s-expression.'
    encodeStream = BytesIO()
    _i.transport = encodeStream
    _i.sendEncoded(lst)
    return encodeStream.getvalue()

def decode(st):
    if False:
        print('Hello World!')
    '\n    Decode a banana-encoded string.\n    '
    l = []
    _i.expressionReceived = l.append
    try:
        _i.dataReceived(st)
    finally:
        _i.buffer = b''
        del _i.expressionReceived
    return l[0]