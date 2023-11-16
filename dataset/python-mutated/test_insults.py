import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import BLINK, CS_ALTERNATE, CS_ALTERNATE_SPECIAL, CS_DRAWING, CS_UK, CS_US, G0, G1, UNDERLINE, ClientProtocol, ServerProtocol, modes, privateModes
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest

def _getattr(mock, name):
    if False:
        return 10
    return super(Mock, mock).__getattribute__(name)

def occurrences(mock):
    if False:
        return 10
    return _getattr(mock, 'occurrences')

def methods(mock):
    if False:
        for i in range(10):
            print('nop')
    return _getattr(mock, 'methods')

def _append(mock, obj):
    if False:
        while True:
            i = 10
    occurrences(mock).append(obj)
default = object()

def _ecmaCodeTableCoordinate(column, row):
    if False:
        while True:
            i = 10
    '\n    Return the byte in 7- or 8-bit code table identified by C{column}\n    and C{row}.\n\n    "An 8-bit code table consists of 256 positions arranged in 16\n    columns and 16 rows.  The columns and rows are numbered 00 to 15."\n\n    "A 7-bit code table consists of 128 positions arranged in 8\n    columns and 16 rows.  The columns are numbered 00 to 07 and the\n    rows 00 to 15 (see figure 1)."\n\n    p.5 of "Standard ECMA-35: Character Code Structure and Extension\n    Techniques", 6th Edition (December 1994).\n    '
    return bytes(bytearray([column << 4 | row]))

def _makeControlFunctionSymbols(name, colOffset, names, doc):
    if False:
        i = 10
        return i + 15
    attrs = {name: ValueConstant(_ecmaCodeTableCoordinate(i + colOffset, j)) for (j, row) in enumerate(names) for (i, name) in enumerate(row) if name}
    attrs['__doc__'] = doc
    return type(name, (Values,), attrs)
CSFinalByte = _makeControlFunctionSymbols('CSFinalByte', colOffset=4, names=[['ICH', 'DCH', 'HPA'], ['CUU', 'SSE', 'HPR'], ['CUD', 'CPR', 'REP'], ['CUF', 'SU', 'DA'], ['CUB', 'SD', 'VPA'], ['CNL', 'NP', 'VPR'], ['CPL', 'PP', 'HVP'], ['CHA', 'CTC', 'TBC'], ['CUP', 'ECH', 'SM'], ['CHT', 'CVT', 'MC'], ['ED', 'CBT', 'HPB'], ['EL', 'SRS', 'VPB'], ['IL', 'PTX', 'RM'], ['DL', 'SDS', 'SGR'], ['EF', 'SIMD', 'DSR'], ['EA', None, 'DAQ']], doc=textwrap.dedent('\n    Symbolic constants for all control sequence final bytes\n    that do not imply intermediate bytes.  This happens to cover\n    movement control sequences.\n\n    See page 11 of "Standard ECMA 48: Control Functions for Coded\n    Character Sets", 5th Edition (June 1991).\n\n    Each L{ValueConstant} maps a control sequence name to L{bytes}\n    '))
C1SevenBit = _makeControlFunctionSymbols('C1SevenBit', colOffset=4, names=[[None, 'DCS'], [None, 'PU1'], ['BPH', 'PU2'], ['NBH', 'STS'], [None, 'CCH'], ['NEL', 'MW'], ['SSA', 'SPA'], ['ESA', 'EPA'], ['HTS', 'SOS'], ['HTJ', None], ['VTS', 'SCI'], ['PLD', 'CSI'], ['PLU', 'ST'], ['RI', 'OSC'], ['SS2', 'PM'], ['SS3', 'APC']], doc=textwrap.dedent('\n    Symbolic constants for all 7 bit versions of the C1 control functions\n\n    See page 9 "Standard ECMA 48: Control Functions for Coded\n    Character Sets", 5th Edition (June 1991).\n\n    Each L{ValueConstant} maps a control sequence name to L{bytes}\n    '))

class Mock:
    callReturnValue = default

    def __init__(self, methods=None, callReturnValue=default):
        if False:
            print('Hello World!')
        '\n        @param methods: Mapping of names to return values\n        @param callReturnValue: object __call__ should return\n        '
        self.occurrences = []
        if methods is None:
            methods = {}
        self.methods = methods
        if callReturnValue is not default:
            self.callReturnValue = callReturnValue

    def __call__(self, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        returnValue = _getattr(self, 'callReturnValue')
        if returnValue is default:
            returnValue = Mock()
        _append(self, ('__call__', returnValue, a, kw))
        return returnValue

    def __getattribute__(self, name):
        if False:
            while True:
                i = 10
        methods = _getattr(self, 'methods')
        if name in methods:
            attrValue = Mock(callReturnValue=methods[name])
        else:
            attrValue = Mock()
        _append(self, (name, attrValue))
        return attrValue

class MockMixin:

    def assertCall(self, occurrence, methodName, expectedPositionalArgs=(), expectedKeywordArgs={}):
        if False:
            i = 10
            return i + 15
        (attr, mock) = occurrence
        self.assertEqual(attr, methodName)
        self.assertEqual(len(occurrences(mock)), 1)
        [(call, result, args, kw)] = occurrences(mock)
        self.assertEqual(call, '__call__')
        self.assertEqual(args, expectedPositionalArgs)
        self.assertEqual(kw, expectedKeywordArgs)
        return result
_byteGroupingTestTemplate = 'def testByte%(groupName)s(self):\n    transport = StringTransport()\n    proto = Mock()\n    parser = self.protocolFactory(lambda: proto)\n    parser.factory = self\n    parser.makeConnection(transport)\n\n    bytes = self.TEST_BYTES\n    while bytes:\n        chunk = bytes[:%(bytesPer)d]\n        bytes = bytes[%(bytesPer)d:]\n        parser.dataReceived(chunk)\n\n    self.verifyResults(transport, proto, parser)\n'

class ByteGroupingsMixin(MockMixin):
    protocolFactory: Optional[Type[Protocol]] = None
    for (word, n) in [('Pairs', 2), ('Triples', 3), ('Quads', 4), ('Quints', 5), ('Sexes', 6)]:
        exec(_byteGroupingTestTemplate % {'groupName': word, 'bytesPer': n})
    del word, n

    def verifyResults(self, transport, proto, parser):
        if False:
            while True:
                i = 10
        result = self.assertCall(occurrences(proto).pop(0), 'makeConnection', (parser,))
        self.assertEqual(occurrences(result), [])
del _byteGroupingTestTemplate

class ServerArrowKeysTests(ByteGroupingsMixin, unittest.TestCase):
    protocolFactory = ServerProtocol
    TEST_BYTES = b'\x1b[A\x1b[B\x1b[C\x1b[D'

    def verifyResults(self, transport, proto, parser):
        if False:
            i = 10
            return i + 15
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for arrow in (parser.UP_ARROW, parser.DOWN_ARROW, parser.RIGHT_ARROW, parser.LEFT_ARROW):
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (arrow, None))
            self.assertEqual(occurrences(result), [])
        self.assertFalse(occurrences(proto))

class PrintableCharactersTests(ByteGroupingsMixin, unittest.TestCase):
    protocolFactory = ServerProtocol
    TEST_BYTES = b'abc123ABC!@#\x1ba\x1bb\x1bc\x1b1\x1b2\x1b3'

    def verifyResults(self, transport, proto, parser):
        if False:
            while True:
                i = 10
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for char in iterbytes(b'abc123ABC!@#'):
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (char, None))
            self.assertEqual(occurrences(result), [])
        for char in iterbytes(b'abc123'):
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (char, parser.ALT))
            self.assertEqual(occurrences(result), [])
        occs = occurrences(proto)
        self.assertFalse(occs, f'{occs!r} should have been []')

class ServerFunctionKeysTests(ByteGroupingsMixin, unittest.TestCase):
    """Test for parsing and dispatching function keys (F1 - F12)"""
    protocolFactory = ServerProtocol
    byteList = []
    for byteCodes in (b'OP', b'OQ', b'OR', b'OS', b'15~', b'17~', b'18~', b'19~', b'20~', b'21~', b'23~', b'24~'):
        byteList.append(b'\x1b[' + byteCodes)
    TEST_BYTES = b''.join(byteList)
    del byteList, byteCodes

    def verifyResults(self, transport, proto, parser):
        if False:
            i = 10
            return i + 15
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for funcNum in range(1, 13):
            funcArg = getattr(parser, 'F%d' % (funcNum,))
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (funcArg, None))
            self.assertEqual(occurrences(result), [])
        self.assertFalse(occurrences(proto))

class ClientCursorMovementTests(ByteGroupingsMixin, unittest.TestCase):
    protocolFactory = ClientProtocol
    d2 = b'\x1b[2B'
    r4 = b'\x1b[4C'
    u1 = b'\x1b[A'
    l2 = b'\x1b[2D'
    TEST_BYTES = d2 + r4 + u1 + l2 + u1 + l2
    del d2, r4, u1, l2

    def verifyResults(self, transport, proto, parser):
        if False:
            while True:
                i = 10
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for (method, count) in [('Down', 2), ('Forward', 4), ('Up', 1), ('Backward', 2), ('Up', 1), ('Backward', 2)]:
            result = self.assertCall(occurrences(proto).pop(0), 'cursor' + method, (count,))
            self.assertEqual(occurrences(result), [])
        self.assertFalse(occurrences(proto))

class ClientControlSequencesTests(unittest.TestCase, MockMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.transport = StringTransport()
        self.proto = Mock()
        self.parser = ClientProtocol(lambda : self.proto)
        self.parser.factory = self
        self.parser.makeConnection(self.transport)
        result = self.assertCall(occurrences(self.proto).pop(0), 'makeConnection', (self.parser,))
        self.assertFalse(occurrences(result))

    def testSimpleCardinals(self):
        if False:
            return 10
        self.parser.dataReceived(b''.join((b'\x1b[' + n + ch for ch in iterbytes(b'BACD') for n in (b'', b'2', b'20', b'200'))))
        occs = occurrences(self.proto)
        for meth in ('Down', 'Up', 'Forward', 'Backward'):
            for count in (1, 2, 20, 200):
                result = self.assertCall(occs.pop(0), 'cursor' + meth, (count,))
                self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testScrollRegion(self):
        if False:
            return 10
        self.parser.dataReceived(b'\x1b[5;22r\x1b[r')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'setScrollRegion', (5, 22))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'setScrollRegion', (None, None))
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testHeightAndWidth(self):
        if False:
            i = 10
            return i + 15
        self.parser.dataReceived(b'\x1b#3\x1b#4\x1b#5\x1b#6')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'doubleHeightLine', (True,))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'doubleHeightLine', (False,))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'singleWidthLine')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'doubleWidthLine')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testCharacterSet(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.dataReceived(b''.join([b''.join([b'\x1b' + g + n for n in iterbytes(b'AB012')]) for g in iterbytes(b'()')]))
        occs = occurrences(self.proto)
        for which in (G0, G1):
            for charset in (CS_UK, CS_US, CS_DRAWING, CS_ALTERNATE, CS_ALTERNATE_SPECIAL):
                result = self.assertCall(occs.pop(0), 'selectCharacterSet', (charset, which))
                self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testShifting(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.dataReceived(b'\x15\x14')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'shiftIn')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'shiftOut')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testSingleShifts(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.dataReceived(b'\x1bN\x1bO')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'singleShift2')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'singleShift3')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testKeypadMode(self):
        if False:
            while True:
                i = 10
        self.parser.dataReceived(b'\x1b=\x1b>')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'applicationKeypadMode')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'numericKeypadMode')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testCursor(self):
        if False:
            print('Hello World!')
        self.parser.dataReceived(b'\x1b7\x1b8')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'saveCursor')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'restoreCursor')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testReset(self):
        if False:
            i = 10
            return i + 15
        self.parser.dataReceived(b'\x1bc')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'reset')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testIndex(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.dataReceived(b'\x1bD\x1bM\x1bE')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'index')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'reverseIndex')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'nextLine')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testModes(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.dataReceived(b'\x1b[' + b';'.join((b'%d' % (m,) for m in [modes.KAM, modes.IRM, modes.LNM])) + b'h')
        self.parser.dataReceived(b'\x1b[' + b';'.join((b'%d' % (m,) for m in [modes.KAM, modes.IRM, modes.LNM])) + b'l')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'setModes', ([modes.KAM, modes.IRM, modes.LNM],))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'resetModes', ([modes.KAM, modes.IRM, modes.LNM],))
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testErasure(self):
        if False:
            print('Hello World!')
        self.parser.dataReceived(b'\x1b[K\x1b[1K\x1b[2K\x1b[J\x1b[1J\x1b[2J\x1b[3P')
        occs = occurrences(self.proto)
        for meth in ('eraseToLineEnd', 'eraseToLineBeginning', 'eraseLine', 'eraseToDisplayEnd', 'eraseToDisplayBeginning', 'eraseDisplay'):
            result = self.assertCall(occs.pop(0), meth)
            self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'deleteCharacter', (3,))
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testLineDeletion(self):
        if False:
            return 10
        self.parser.dataReceived(b'\x1b[M\x1b[3M')
        occs = occurrences(self.proto)
        for arg in (1, 3):
            result = self.assertCall(occs.pop(0), 'deleteLine', (arg,))
            self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testLineInsertion(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.dataReceived(b'\x1b[L\x1b[3L')
        occs = occurrences(self.proto)
        for arg in (1, 3):
            result = self.assertCall(occs.pop(0), 'insertLine', (arg,))
            self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testCursorPosition(self):
        if False:
            while True:
                i = 10
        methods(self.proto)['reportCursorPosition'] = (6, 7)
        self.parser.dataReceived(b'\x1b[6n')
        self.assertEqual(self.transport.value(), b'\x1b[7;8R')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'reportCursorPosition')
        self.assertEqual(result, (6, 7))

    def test_applicationDataBytes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Contiguous non-control bytes are passed to a single call to the\n        C{write} method of the terminal to which the L{ClientProtocol} is\n        connected.\n        '
        occs = occurrences(self.proto)
        self.parser.dataReceived(b'a')
        self.assertCall(occs.pop(0), 'write', (b'a',))
        self.parser.dataReceived(b'bc')
        self.assertCall(occs.pop(0), 'write', (b'bc',))

    def _applicationDataTest(self, data, calls):
        if False:
            print('Hello World!')
        occs = occurrences(self.proto)
        self.parser.dataReceived(data)
        while calls:
            self.assertCall(occs.pop(0), *calls.pop(0))
        self.assertFalse(occs, f'No other calls should happen: {occs!r}')

    def test_shiftInAfterApplicationData(self):
        if False:
            print('Hello World!')
        "\n        Application data bytes followed by a shift-in command are passed to a\n        call to C{write} before the terminal's C{shiftIn} method is called.\n        "
        self._applicationDataTest(b'ab\x15', [('write', (b'ab',)), ('shiftIn',)])

    def test_shiftOutAfterApplicationData(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Application data bytes followed by a shift-out command are passed to a\n        call to C{write} before the terminal's C{shiftOut} method is called.\n        "
        self._applicationDataTest(b'ab\x14', [('write', (b'ab',)), ('shiftOut',)])

    def test_cursorBackwardAfterApplicationData(self):
        if False:
            return 10
        "\n        Application data bytes followed by a cursor-backward command are passed\n        to a call to C{write} before the terminal's C{cursorBackward} method is\n        called.\n        "
        self._applicationDataTest(b'ab\x08', [('write', (b'ab',)), ('cursorBackward',)])

    def test_escapeAfterApplicationData(self):
        if False:
            return 10
        "\n        Application data bytes followed by an escape character are passed to a\n        call to C{write} before the terminal's handler method for the escape is\n        called.\n        "
        self._applicationDataTest(b'ab\x1bD', [('write', (b'ab',)), ('index',)])
        self._applicationDataTest(b'ab\x1b[4h', [('write', (b'ab',)), ('setModes', ([4],))])

class ServerProtocolOutputTests(unittest.TestCase):
    """
    Tests for the bytes L{ServerProtocol} writes to its transport when its
    methods are called.
    """
    ESC = _ecmaCodeTableCoordinate(1, 11)
    CSI = ESC + _ecmaCodeTableCoordinate(5, 11)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.protocol = ServerProtocol()
        self.transport = StringTransport()
        self.protocol.makeConnection(self.transport)

    def test_cursorUp(self):
        if False:
            return 10
        '\n        L{ServerProtocol.cursorUp} writes the control sequence\n        ending with L{CSFinalByte.CUU} to its transport.\n        '
        self.protocol.cursorUp(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUU.value)

    def test_cursorDown(self):
        if False:
            print('Hello World!')
        '\n        L{ServerProtocol.cursorDown} writes the control sequence\n        ending with L{CSFinalByte.CUD} to its transport.\n        '
        self.protocol.cursorDown(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUD.value)

    def test_cursorForward(self):
        if False:
            return 10
        '\n        L{ServerProtocol.cursorForward} writes the control sequence\n        ending with L{CSFinalByte.CUF} to its transport.\n        '
        self.protocol.cursorForward(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUF.value)

    def test_cursorBackward(self):
        if False:
            while True:
                i = 10
        '\n        L{ServerProtocol.cursorBackward} writes the control sequence\n        ending with L{CSFinalByte.CUB} to its transport.\n        '
        self.protocol.cursorBackward(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUB.value)

    def test_cursorPosition(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ServerProtocol.cursorPosition} writes a control sequence\n        ending with L{CSFinalByte.CUP} and containing the expected\n        coordinates to its transport.\n        '
        self.protocol.cursorPosition(0, 0)
        self.assertEqual(self.transport.value(), self.CSI + b'1;1' + CSFinalByte.CUP.value)

    def test_cursorHome(self):
        if False:
            return 10
        '\n        L{ServerProtocol.cursorHome} writes a control sequence ending\n        with L{CSFinalByte.CUP} and no parameters, so that the client\n        defaults to (1, 1).\n        '
        self.protocol.cursorHome()
        self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.CUP.value)

    def test_index(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ServerProtocol.index} writes the control sequence ending in\n        the 8-bit code table coordinates 4, 4.\n\n        Note that ECMA48 5th Edition removes C{IND}.\n        '
        self.protocol.index()
        self.assertEqual(self.transport.value(), self.ESC + _ecmaCodeTableCoordinate(4, 4))

    def test_reverseIndex(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ServerProtocol.reverseIndex} writes the control sequence\n        ending in the L{C1SevenBit.RI}.\n        '
        self.protocol.reverseIndex()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.RI.value)

    def test_nextLine(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ServerProtocol.nextLine} writes C{"\r\n"} to its transport.\n        '
        self.protocol.nextLine()
        self.assertEqual(self.transport.value(), b'\r\n')

    def test_setModes(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ServerProtocol.setModes} writes a control sequence\n        containing the requested modes and ending in the\n        L{CSFinalByte.SM}.\n        '
        modesToSet = [modes.KAM, modes.IRM, modes.LNM]
        self.protocol.setModes(modesToSet)
        self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in modesToSet)) + CSFinalByte.SM.value)

    def test_setPrivateModes(self):
        if False:
            while True:
                i = 10
        '\n        L{ServerProtocol.setPrivatesModes} writes a control sequence\n        containing the requested private modes and ending in the\n        L{CSFinalByte.SM}.\n        '
        privateModesToSet = [privateModes.ERROR, privateModes.COLUMN, privateModes.ORIGIN]
        self.protocol.setModes(privateModesToSet)
        self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in privateModesToSet)) + CSFinalByte.SM.value)

    def test_resetModes(self):
        if False:
            while True:
                i = 10
        '\n        L{ServerProtocol.resetModes} writes the control sequence\n        ending in the L{CSFinalByte.RM}.\n        '
        modesToSet = [modes.KAM, modes.IRM, modes.LNM]
        self.protocol.resetModes(modesToSet)
        self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in modesToSet)) + CSFinalByte.RM.value)

    def test_singleShift2(self):
        if False:
            print('Hello World!')
        '\n        L{ServerProtocol.singleShift2} writes an escape sequence\n        followed by L{C1SevenBit.SS2}\n        '
        self.protocol.singleShift2()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.SS2.value)

    def test_singleShift3(self):
        if False:
            return 10
        '\n        L{ServerProtocol.singleShift3} writes an escape sequence\n        followed by L{C1SevenBit.SS3}\n        '
        self.protocol.singleShift3()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.SS3.value)

    def test_selectGraphicRendition(self):
        if False:
            return 10
        '\n        L{ServerProtocol.selectGraphicRendition} writes a control\n        sequence containing the requested attributes and ending with\n        L{CSFinalByte.SGR}\n        '
        self.protocol.selectGraphicRendition(str(BLINK), str(UNDERLINE))
        self.assertEqual(self.transport.value(), self.CSI + b'%d;%d' % (BLINK, UNDERLINE) + CSFinalByte.SGR.value)

    def test_horizontalTabulationSet(self):
        if False:
            while True:
                i = 10
        '\n        L{ServerProtocol.horizontalTabulationSet} writes the escape\n        sequence ending in L{C1SevenBit.HTS}\n        '
        self.protocol.horizontalTabulationSet()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.HTS.value)

    def test_eraseToLineEnd(self):
        if False:
            return 10
        "\n        L{ServerProtocol.eraseToLineEnd} writes the control sequence\n        sequence ending in L{CSFinalByte.EL} and no parameters,\n        forcing the client to default to 0 (from the active present\n        position's current location to the end of the line.)\n        "
        self.protocol.eraseToLineEnd()
        self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.EL.value)

    def test_eraseToLineBeginning(self):
        if False:
            while True:
                i = 10
        "\n        L{ServerProtocol.eraseToLineBeginning} writes the control\n        sequence sequence ending in L{CSFinalByte.EL} and a parameter\n        of 1 (from the beginning of the line up to and include the\n        active present position's current location.)\n        "
        self.protocol.eraseToLineBeginning()
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.EL.value)

    def test_eraseLine(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ServerProtocol.eraseLine} writes the control\n        sequence sequence ending in L{CSFinalByte.EL} and a parameter\n        of 2 (the entire line.)\n        '
        self.protocol.eraseLine()
        self.assertEqual(self.transport.value(), self.CSI + b'2' + CSFinalByte.EL.value)

    def test_eraseToDisplayEnd(self):
        if False:
            return 10
        "\n        L{ServerProtocol.eraseToDisplayEnd} writes the control\n        sequence sequence ending in L{CSFinalByte.ED} and no parameters,\n        forcing the client to default to 0 (from the active present\n        position's current location to the end of the page.)\n        "
        self.protocol.eraseToDisplayEnd()
        self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.ED.value)

    def test_eraseToDisplayBeginning(self):
        if False:
            i = 10
            return i + 15
        "\n        L{ServerProtocol.eraseToDisplayBeginning} writes the control\n        sequence sequence ending in L{CSFinalByte.ED} a parameter of 1\n        (from the beginning of the page up to and include the active\n        present position's current location.)\n        "
        self.protocol.eraseToDisplayBeginning()
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.ED.value)

    def test_eraseToDisplay(self):
        if False:
            print('Hello World!')
        '\n        L{ServerProtocol.eraseDisplay} writes the control sequence\n        sequence ending in L{CSFinalByte.ED} a parameter of 2 (the\n        entire page)\n        '
        self.protocol.eraseDisplay()
        self.assertEqual(self.transport.value(), self.CSI + b'2' + CSFinalByte.ED.value)

    def test_deleteCharacter(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ServerProtocol.deleteCharacter} writes the control sequence\n        containing the number of characters to delete and ending in\n        L{CSFinalByte.DCH}\n        '
        self.protocol.deleteCharacter(4)
        self.assertEqual(self.transport.value(), self.CSI + b'4' + CSFinalByte.DCH.value)

    def test_insertLine(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ServerProtocol.insertLine} writes the control sequence\n        containing the number of lines to insert and ending in\n        L{CSFinalByte.IL}\n        '
        self.protocol.insertLine(5)
        self.assertEqual(self.transport.value(), self.CSI + b'5' + CSFinalByte.IL.value)

    def test_deleteLine(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ServerProtocol.deleteLine} writes the control sequence\n        containing the number of lines to delete and ending in\n        L{CSFinalByte.DL}\n        '
        self.protocol.deleteLine(6)
        self.assertEqual(self.transport.value(), self.CSI + b'6' + CSFinalByte.DL.value)

    def test_setScrollRegionNoArgs(self):
        if False:
            print('Hello World!')
        "\n        With no arguments, L{ServerProtocol.setScrollRegion} writes a\n        control sequence with no parameters, but a parameter\n        separator, and ending in C{b'r'}.\n        "
        self.protocol.setScrollRegion()
        self.assertEqual(self.transport.value(), self.CSI + b';' + b'r')

    def test_setScrollRegionJustFirst(self):
        if False:
            while True:
                i = 10
        "\n        With just a value for its C{first} argument,\n        L{ServerProtocol.setScrollRegion} writes a control sequence with\n        that parameter, a parameter separator, and finally a C{b'r'}.\n        "
        self.protocol.setScrollRegion(first=1)
        self.assertEqual(self.transport.value(), self.CSI + b'1;' + b'r')

    def test_setScrollRegionJustLast(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        With just a value for its C{last} argument,\n        L{ServerProtocol.setScrollRegion} writes a control sequence with\n        a parameter separator, that parameter, and finally a C{b'r'}.\n        "
        self.protocol.setScrollRegion(last=1)
        self.assertEqual(self.transport.value(), self.CSI + b';1' + b'r')

    def test_setScrollRegionFirstAndLast(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        When given both C{first} and C{last}\n        L{ServerProtocol.setScrollRegion} writes a control sequence with\n        the first parameter, a parameter separator, the last\n        parameter, and finally a C{b'r'}.\n        "
        self.protocol.setScrollRegion(first=1, last=2)
        self.assertEqual(self.transport.value(), self.CSI + b'1;2' + b'r')

    def test_reportCursorPosition(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ServerProtocol.reportCursorPosition} writes a control\n        sequence ending in L{CSFinalByte.DSR} with a parameter of 6\n        (the Device Status Report returns the current active\n        position.)\n        '
        self.protocol.reportCursorPosition()
        self.assertEqual(self.transport.value(), self.CSI + b'6' + CSFinalByte.DSR.value)