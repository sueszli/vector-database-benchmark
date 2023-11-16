import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
_maxint = 9223372036854775807

class MathTests(TestCase):

    def test_int2b128(self):
        if False:
            i = 10
            return i + 15
        funkylist = list(range(0, 100)) + list(range(1000, 1100)) + list(range(1000000, 1000100)) + [1024 ** 10]
        for i in funkylist:
            x = BytesIO()
            banana.int2b128(i, x.write)
            v = x.getvalue()
            y = banana.b1282int(v)
            self.assertEqual(y, i)

def selectDialect(protocol, dialect):
    if False:
        print('Hello World!')
    '\n    Dictate a Banana dialect to use.\n\n    @param protocol: A L{banana.Banana} instance which has not yet had a\n        dialect negotiated.\n\n    @param dialect: A L{bytes} instance naming a Banana dialect to select.\n    '
    protocol._selectDialect(dialect)

def encode(bananaFactory, obj):
    if False:
        print('Hello World!')
    '\n    Banana encode an object using L{banana.Banana.sendEncoded}.\n\n    @param bananaFactory: A no-argument callable which will return a new,\n        unconnected protocol instance to use to do the encoding (this should\n        most likely be a L{banana.Banana} instance).\n\n    @param obj: The object to encode.\n    @type obj: Any type supported by Banana.\n\n    @return: A L{bytes} instance giving the encoded form of C{obj}.\n    '
    transport = StringTransport()
    banana = bananaFactory()
    banana.makeConnection(transport)
    transport.clear()
    banana.sendEncoded(obj)
    return transport.value()

class BananaTestBase(TestCase):
    """
    The base for test classes. It defines commonly used things and sets up a
    connection for testing.
    """
    encClass = banana.Banana

    def setUp(self):
        if False:
            print('Hello World!')
        self.io = BytesIO()
        self.enc = self.encClass()
        self.enc.makeConnection(protocol.FileWrapper(self.io))
        selectDialect(self.enc, b'none')
        self.enc.expressionReceived = self.putResult
        self.encode = partial(encode, self.encClass)

    def putResult(self, result):
        if False:
            for i in range(10):
                print('nop')
        '\n        Store an expression received by C{self.enc}.\n\n        @param result: The object that was received.\n        @type result: Any type supported by Banana.\n        '
        self.result = result

    def tearDown(self):
        if False:
            return 10
        self.enc.connectionLost(failure.Failure(main.CONNECTION_DONE))
        del self.enc

class BananaTests(BananaTestBase):
    """
    General banana tests.
    """

    def test_string(self):
        if False:
            while True:
                i = 10
        self.enc.sendEncoded(b'hello')
        self.enc.dataReceived(self.io.getvalue())
        assert self.result == b'hello'

    def test_unsupportedUnicode(self):
        if False:
            return 10
        '\n        Banana does not support unicode.  ``Banana.sendEncoded`` raises\n        ``BananaError`` if called with an instance of ``unicode``.\n        '
        self._unsupportedTypeTest('hello', 'builtins.str')

    def test_unsupportedBuiltinType(self):
        if False:
            print('Hello World!')
        '\n        Banana does not support arbitrary builtin types like L{type}.\n        L{banana.Banana.sendEncoded} raises L{banana.BananaError} if called\n        with an instance of L{type}.\n        '
        self._unsupportedTypeTest(type, 'builtins.type')

    def test_unsupportedUserType(self):
        if False:
            while True:
                i = 10
        '\n        Banana does not support arbitrary user-defined types (such as those\n        defined with the ``class`` statement).  ``Banana.sendEncoded`` raises\n        ``BananaError`` if called with an instance of such a type.\n        '
        self._unsupportedTypeTest(MathTests(), __name__ + '.MathTests')

    def _unsupportedTypeTest(self, obj, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that L{banana.Banana.sendEncoded} raises L{banana.BananaError}\n        if called with the given object.\n\n        @param obj: Some object that Banana does not support.\n        @param name: The name of the type of the object.\n\n        @raise: The failure exception is raised if L{Banana.sendEncoded} does\n            not raise L{banana.BananaError} or if the message associated with the\n            exception is not formatted to include the type of the unsupported\n            object.\n        '
        exc = self.assertRaises(banana.BananaError, self.enc.sendEncoded, obj)
        self.assertIn(f'Banana cannot send {name} objects', str(exc))

    def test_int(self):
        if False:
            while True:
                i = 10
        '\n        A positive integer less than 2 ** 32 should round-trip through\n        banana without changing value and should come out represented\n        as an C{int} (regardless of the type which was encoded).\n        '
        self.enc.sendEncoded(10151)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, 10151)
        self.assertIsInstance(self.result, int)

    def _getSmallest(self):
        if False:
            for i in range(10):
                print('nop')
        bytes = self.enc.prefixLimit
        bits = bytes * 7
        largest = 2 ** bits - 1
        smallest = largest + 1
        return smallest

    def test_encodeTooLargeLong(self):
        if False:
            print('Hello World!')
        '\n        Test that a long above the implementation-specific limit is rejected\n        as too large to be encoded.\n        '
        smallest = self._getSmallest()
        self.assertRaises(banana.BananaError, self.enc.sendEncoded, smallest)

    def test_decodeTooLargeLong(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that a long above the implementation specific limit is rejected\n        as too large to be decoded.\n        '
        smallest = self._getSmallest()
        self.enc.setPrefixLimit(self.enc.prefixLimit * 2)
        self.enc.sendEncoded(smallest)
        encoded = self.io.getvalue()
        self.io.truncate(0)
        self.enc.setPrefixLimit(self.enc.prefixLimit // 2)
        self.assertRaises(banana.BananaError, self.enc.dataReceived, encoded)

    def _getLargest(self):
        if False:
            while True:
                i = 10
        return -self._getSmallest()

    def test_encodeTooSmallLong(self):
        if False:
            while True:
                i = 10
        '\n        Test that a negative long below the implementation-specific limit is\n        rejected as too small to be encoded.\n        '
        largest = self._getLargest()
        self.assertRaises(banana.BananaError, self.enc.sendEncoded, largest)

    def test_decodeTooSmallLong(self):
        if False:
            return 10
        '\n        Test that a negative long below the implementation specific limit is\n        rejected as too small to be decoded.\n        '
        largest = self._getLargest()
        self.enc.setPrefixLimit(self.enc.prefixLimit * 2)
        self.enc.sendEncoded(largest)
        encoded = self.io.getvalue()
        self.io.truncate(0)
        self.enc.setPrefixLimit(self.enc.prefixLimit // 2)
        self.assertRaises(banana.BananaError, self.enc.dataReceived, encoded)

    def test_integer(self):
        if False:
            for i in range(10):
                print('nop')
        self.enc.sendEncoded(1015)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, 1015)

    def test_negative(self):
        if False:
            for i in range(10):
                print('nop')
        self.enc.sendEncoded(-1015)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, -1015)

    def test_float(self):
        if False:
            i = 10
            return i + 15
        self.enc.sendEncoded(1015.0)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, 1015.0)

    def test_list(self):
        if False:
            for i in range(10):
                print('nop')
        foo = [1, 2, [3, 4], [30.5, 40.2], 5, [b'six', b'seven', [b'eight', 9]], [10], []]
        self.enc.sendEncoded(foo)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, foo)

    def test_partial(self):
        if False:
            print('Hello World!')
        '\n        Test feeding the data byte per byte to the receiver. Normally\n        data is not split.\n        '
        foo = [1, 2, [3, 4], [30.5, 40.2], 5, [b'six', b'seven', [b'eight', 9]], [10], sys.maxsize * 3, sys.maxsize * 2, sys.maxsize * -2]
        self.enc.sendEncoded(foo)
        self.feed(self.io.getvalue())
        self.assertEqual(self.result, foo)

    def feed(self, data):
        if False:
            while True:
                i = 10
        '\n        Feed the data byte per byte to the receiver.\n\n        @param data: The bytes to deliver.\n        @type data: L{bytes}\n        '
        for byte in iterbytes(data):
            self.enc.dataReceived(byte)

    def test_oversizedList(self):
        if False:
            for i in range(10):
                print('nop')
        data = b'\x02\x01\x01\x01\x01\x80'
        self.assertRaises(banana.BananaError, self.feed, data)

    def test_oversizedString(self):
        if False:
            print('Hello World!')
        data = b'\x02\x01\x01\x01\x01\x82'
        self.assertRaises(banana.BananaError, self.feed, data)

    def test_crashString(self):
        if False:
            for i in range(10):
                print('nop')
        crashString = b'\x00\x00\x00\x00\x04\x80'
        try:
            self.enc.dataReceived(crashString)
        except banana.BananaError:
            pass

    def test_crashNegativeLong(self):
        if False:
            print('Hello World!')
        self.enc.sendEncoded(-2147483648)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, -2147483648)

    def test_sizedIntegerTypes(self):
        if False:
            while True:
                i = 10
        '\n        Test that integers below the maximum C{INT} token size cutoff are\n        serialized as C{INT} or C{NEG} and that larger integers are\n        serialized as C{LONGINT} or C{LONGNEG}.\n        '
        baseIntIn = +2147483647
        baseNegIn = -2147483648
        baseIntOut = b'\x7f\x7f\x7f\x07\x81'
        self.assertEqual(self.encode(baseIntIn - 2), b'}' + baseIntOut)
        self.assertEqual(self.encode(baseIntIn - 1), b'~' + baseIntOut)
        self.assertEqual(self.encode(baseIntIn - 0), b'\x7f' + baseIntOut)
        baseLongIntOut = b'\x00\x00\x00\x08\x85'
        self.assertEqual(self.encode(baseIntIn + 1), b'\x00' + baseLongIntOut)
        self.assertEqual(self.encode(baseIntIn + 2), b'\x01' + baseLongIntOut)
        self.assertEqual(self.encode(baseIntIn + 3), b'\x02' + baseLongIntOut)
        baseNegOut = b'\x7f\x7f\x7f\x07\x83'
        self.assertEqual(self.encode(baseNegIn + 2), b'~' + baseNegOut)
        self.assertEqual(self.encode(baseNegIn + 1), b'\x7f' + baseNegOut)
        self.assertEqual(self.encode(baseNegIn + 0), b'\x00\x00\x00\x00\x08\x83')
        baseLongNegOut = b'\x00\x00\x00\x08\x86'
        self.assertEqual(self.encode(baseNegIn - 1), b'\x01' + baseLongNegOut)
        self.assertEqual(self.encode(baseNegIn - 2), b'\x02' + baseLongNegOut)
        self.assertEqual(self.encode(baseNegIn - 3), b'\x03' + baseLongNegOut)

class DialectTests(BananaTestBase):
    """
    Tests for Banana's handling of dialects.
    """
    vocab = b'remote'
    legalPbItem = bytes((banana.Banana.outgoingVocabulary[vocab],)) + banana.VOCAB
    illegalPbItem = bytes((122,)) + banana.VOCAB

    def test_dialectNotSet(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If no dialect has been selected and a PB VOCAB item is received,\n        L{NotImplementedError} is raised.\n        '
        self.assertRaises(NotImplementedError, self.enc.dataReceived, self.legalPbItem)

    def test_receivePb(self):
        if False:
            print('Hello World!')
        '\n        If the PB dialect has been selected, a PB VOCAB item is accepted.\n        '
        selectDialect(self.enc, b'pb')
        self.enc.dataReceived(self.legalPbItem)
        self.assertEqual(self.result, self.vocab)

    def test_receiveIllegalPb(self):
        if False:
            i = 10
            return i + 15
        '\n        If the PB dialect has been selected and an unrecognized PB VOCAB item\n        is received, L{banana.Banana.dataReceived} raises L{KeyError}.\n        '
        selectDialect(self.enc, b'pb')
        self.assertRaises(KeyError, self.enc.dataReceived, self.illegalPbItem)

    def test_sendPb(self):
        if False:
            i = 10
            return i + 15
        '\n        if pb dialect is selected, the sender must be able to send things in\n        that dialect.\n        '
        selectDialect(self.enc, b'pb')
        self.enc.sendEncoded(self.vocab)
        self.assertEqual(self.legalPbItem, self.io.getvalue())

class GlobalCoderTests(TestCase):
    """
    Tests for the free functions L{banana.encode} and L{banana.decode}.
    """

    def test_statelessDecode(self):
        if False:
            while True:
                i = 10
        '\n        Calls to L{banana.decode} are independent of each other.\n        '
        undecodable = b'\x7f' * 65 + b'\x85'
        self.assertRaises(banana.BananaError, banana.decode, undecodable)
        decodable = b'\x01\x81'
        self.assertEqual(banana.decode(decodable), 1)