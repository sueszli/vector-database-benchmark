"""
VT102 and VT220 terminal manipulation.

@author: Jp Calderone
"""
from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString

class ITerminalProtocol(Interface):

    def makeConnection(transport):
        if False:
            return 10
        '\n        Called with an L{ITerminalTransport} when a connection is established.\n        '

    def keystrokeReceived(keyID, modifier):
        if False:
            for i in range(10):
                print('nop')
        '\n        A keystroke was received.\n\n        Each keystroke corresponds to one invocation of this method.\n        keyID is a string identifier for that key.  Printable characters\n        are represented by themselves.  Control keys, such as arrows and\n        function keys, are represented with symbolic constants on\n        L{ServerProtocol}.\n        '

    def terminalSize(width, height):
        if False:
            print('Hello World!')
        '\n        Called to indicate the size of the terminal.\n\n        A terminal of 80x24 should be assumed if this method is not\n        called.  This method might not be called for real terminals.\n        '

    def unhandledControlSequence(seq):
        if False:
            i = 10
            return i + 15
        '\n        Called when an unsupported control sequence is received.\n\n        @type seq: L{str}\n        @param seq: The whole control sequence which could not be interpreted.\n        '

    def connectionLost(reason):
        if False:
            return 10
        '\n        Called when the connection has been lost.\n\n        reason is a Failure describing why.\n        '

@implementer(ITerminalProtocol)
class TerminalProtocol:

    def makeConnection(self, terminal):
        if False:
            for i in range(10):
                print('nop')
        self.terminal = terminal
        self.connectionMade()

    def connectionMade(self):
        if False:
            print('Hello World!')
        '\n        Called after a connection has been established.\n        '

    def keystrokeReceived(self, keyID, modifier):
        if False:
            return 10
        pass

    def terminalSize(self, width, height):
        if False:
            print('Hello World!')
        pass

    def unhandledControlSequence(self, seq):
        if False:
            print('Hello World!')
        pass

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        pass

class ITerminalTransport(iinternet.ITransport):

    def cursorUp(n=1):
        if False:
            while True:
                i = 10
        '\n        Move the cursor up n lines.\n        '

    def cursorDown(n=1):
        if False:
            i = 10
            return i + 15
        '\n        Move the cursor down n lines.\n        '

    def cursorForward(n=1):
        if False:
            print('Hello World!')
        '\n        Move the cursor right n columns.\n        '

    def cursorBackward(n=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Move the cursor left n columns.\n        '

    def cursorPosition(column, line):
        if False:
            return 10
        '\n        Move the cursor to the given line and column.\n        '

    def cursorHome():
        if False:
            while True:
                i = 10
        '\n        Move the cursor home.\n        '

    def index():
        if False:
            print('Hello World!')
        '\n        Move the cursor down one line, performing scrolling if necessary.\n        '

    def reverseIndex():
        if False:
            for i in range(10):
                print('nop')
        '\n        Move the cursor up one line, performing scrolling if necessary.\n        '

    def nextLine():
        if False:
            return 10
        '\n        Move the cursor to the first position on the next line, performing scrolling if necessary.\n        '

    def saveCursor():
        if False:
            for i in range(10):
                print('nop')
        '\n        Save the cursor position, character attribute, character set, and origin mode selection.\n        '

    def restoreCursor():
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore the previously saved cursor position, character attribute, character set, and origin mode selection.\n\n        If no cursor state was previously saved, move the cursor to the home position.\n        '

    def setModes(modes):
        if False:
            i = 10
            return i + 15
        '\n        Set the given modes on the terminal.\n        '

    def resetModes(mode):
        if False:
            return 10
        '\n        Reset the given modes on the terminal.\n        '

    def setPrivateModes(modes):
        if False:
            return 10
        '\n        Set the given DEC private modes on the terminal.\n        '

    def resetPrivateModes(modes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset the given DEC private modes on the terminal.\n        '

    def applicationKeypadMode():
        if False:
            print('Hello World!')
        '\n        Cause keypad to generate control functions.\n\n        Cursor key mode selects the type of characters generated by cursor keys.\n        '

    def numericKeypadMode():
        if False:
            for i in range(10):
                print('nop')
        '\n        Cause keypad to generate normal characters.\n        '

    def selectCharacterSet(charSet, which):
        if False:
            while True:
                i = 10
        '\n        Select a character set.\n\n        charSet should be one of CS_US, CS_UK, CS_DRAWING, CS_ALTERNATE, or\n        CS_ALTERNATE_SPECIAL.\n\n        which should be one of G0 or G1.\n        '

    def shiftIn():
        if False:
            return 10
        '\n        Activate the G0 character set.\n        '

    def shiftOut():
        if False:
            print('Hello World!')
        '\n        Activate the G1 character set.\n        '

    def singleShift2():
        if False:
            for i in range(10):
                print('nop')
        '\n        Shift to the G2 character set for a single character.\n        '

    def singleShift3():
        if False:
            for i in range(10):
                print('nop')
        '\n        Shift to the G3 character set for a single character.\n        '

    def selectGraphicRendition(*attributes):
        if False:
            i = 10
            return i + 15
        '\n        Enabled one or more character attributes.\n\n        Arguments should be one or more of UNDERLINE, REVERSE_VIDEO, BLINK, or BOLD.\n        NORMAL may also be specified to disable all character attributes.\n        '

    def horizontalTabulationSet():
        if False:
            while True:
                i = 10
        '\n        Set a tab stop at the current cursor position.\n        '

    def tabulationClear():
        if False:
            i = 10
            return i + 15
        '\n        Clear the tab stop at the current cursor position.\n        '

    def tabulationClearAll():
        if False:
            return 10
        '\n        Clear all tab stops.\n        '

    def doubleHeightLine(top=True):
        if False:
            while True:
                i = 10
        '\n        Make the current line the top or bottom half of a double-height, double-width line.\n\n        If top is True, the current line is the top half.  Otherwise, it is the bottom half.\n        '

    def singleWidthLine():
        if False:
            print('Hello World!')
        '\n        Make the current line a single-width, single-height line.\n        '

    def doubleWidthLine():
        if False:
            while True:
                i = 10
        '\n        Make the current line a double-width line.\n        '

    def eraseToLineEnd():
        if False:
            return 10
        '\n        Erase from the cursor to the end of line, including cursor position.\n        '

    def eraseToLineBeginning():
        if False:
            while True:
                i = 10
        '\n        Erase from the cursor to the beginning of the line, including the cursor position.\n        '

    def eraseLine():
        if False:
            while True:
                i = 10
        '\n        Erase the entire cursor line.\n        '

    def eraseToDisplayEnd():
        if False:
            i = 10
            return i + 15
        '\n        Erase from the cursor to the end of the display, including the cursor position.\n        '

    def eraseToDisplayBeginning():
        if False:
            i = 10
            return i + 15
        '\n        Erase from the cursor to the beginning of the display, including the cursor position.\n        '

    def eraseDisplay():
        if False:
            for i in range(10):
                print('nop')
        '\n        Erase the entire display.\n        '

    def deleteCharacter(n=1):
        if False:
            return 10
        '\n        Delete n characters starting at the cursor position.\n\n        Characters to the right of deleted characters are shifted to the left.\n        '

    def insertLine(n=1):
        if False:
            while True:
                i = 10
        '\n        Insert n lines at the cursor position.\n\n        Lines below the cursor are shifted down.  Lines moved past the bottom margin are lost.\n        This command is ignored when the cursor is outside the scroll region.\n        '

    def deleteLine(n=1):
        if False:
            i = 10
            return i + 15
        '\n        Delete n lines starting at the cursor position.\n\n        Lines below the cursor are shifted up.  This command is ignored when the cursor is outside\n        the scroll region.\n        '

    def reportCursorPosition():
        if False:
            print('Hello World!')
        '\n        Return a Deferred that fires with a two-tuple of (x, y) indicating the cursor position.\n        '

    def reset():
        if False:
            return 10
        '\n        Reset the terminal to its initial state.\n        '

    def unhandledControlSequence(seq):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when an unsupported control sequence is received.\n\n        @type seq: L{str}\n        @param seq: The whole control sequence which could not be interpreted.\n        '
CSI = b'\x1b'
CST = {b'~': b'tilde'}

class modes:
    """
    ECMA 48 standardized modes
    """
    KEYBOARD_ACTION = KAM = 2
    INSERTION_REPLACEMENT = IRM = 4
    LINEFEED_NEWLINE = LNM = 20

class privateModes:
    """
    ANSI-Compatible Private Modes
    """
    ERROR = 0
    CURSOR_KEY = 1
    ANSI_VT52 = 2
    COLUMN = 3
    SCROLL = 4
    SCREEN = 5
    ORIGIN = 6
    AUTO_WRAP = 7
    AUTO_REPEAT = 8
    PRINTER_FORM_FEED = 18
    PRINTER_EXTENT = 19
    CURSOR_MODE = 25
CS_US = b'CS_US'
CS_UK = b'CS_UK'
CS_DRAWING = b'CS_DRAWING'
CS_ALTERNATE = b'CS_ALTERNATE'
CS_ALTERNATE_SPECIAL = b'CS_ALTERNATE_SPECIAL'
G0 = b'G0'
G1 = b'G1'
G2 = b'G2'
G3 = b'G3'
NORMAL = 0
BOLD = 1
UNDERLINE = 4
BLINK = 5
REVERSE_VIDEO = 7

class Vector:

    def __init__(self, x, y):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y

def log(s):
    if False:
        return 10
    with open('log', 'a') as f:
        f.write(str(s) + '\n')
_KEY_NAMES = ('UP_ARROW', 'DOWN_ARROW', 'RIGHT_ARROW', 'LEFT_ARROW', 'HOME', 'INSERT', 'DELETE', 'END', 'PGUP', 'PGDN', 'NUMPAD_MIDDLE', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'ALT', 'SHIFT', 'CONTROL')

class _const:
    """
    @ivar name: A string naming this constant
    """

    def __init__(self, name: str) -> None:
        if False:
            while True:
                i = 10
        self.name = name

    def __repr__(self) -> str:
        if False:
            return 10
        return '[' + self.name + ']'

    def __bytes__(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        return ('[' + self.name + ']').encode('ascii')
FUNCTION_KEYS = [_const(_name).__bytes__() for _name in _KEY_NAMES]

@implementer(ITerminalTransport)
class ServerProtocol(protocol.Protocol):
    protocolFactory = None
    terminalProtocol = None
    TAB = b'\t'
    BACKSPACE = b'\x7f'
    lastWrite = b''
    state = b'data'
    termSize = Vector(80, 24)
    cursorPos = Vector(0, 0)
    scrollRegion = None
    factory = None

    def __init__(self, protocolFactory=None, *a, **kw):
        if False:
            i = 10
            return i + 15
        '\n        @param protocolFactory: A callable which will be invoked with\n        *a, **kw and should return an ITerminalProtocol implementor.\n        This will be invoked when a connection to this ServerProtocol\n        is established.\n\n        @param a: Any positional arguments to pass to protocolFactory.\n        @param kw: Any keyword arguments to pass to protocolFactory.\n        '
        if protocolFactory is not None:
            self.protocolFactory = protocolFactory
        self.protocolArgs = a
        self.protocolKwArgs = kw
        self._cursorReports = []

    def getHost(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: ServerProtocol.getHost')

    def getPeer(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Unimplemented: ServerProtocol.getPeer')

    def connectionMade(self):
        if False:
            while True:
                i = 10
        if self.protocolFactory is not None:
            self.terminalProtocol = self.protocolFactory(*self.protocolArgs, **self.protocolKwArgs)
            try:
                factory = self.factory
            except AttributeError:
                pass
            else:
                self.terminalProtocol.factory = factory
            self.terminalProtocol.makeConnection(self)

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        for ch in iterbytes(data):
            if self.state == b'data':
                if ch == b'\x1b':
                    self.state = b'escaped'
                else:
                    self.terminalProtocol.keystrokeReceived(ch, None)
            elif self.state == b'escaped':
                if ch == b'[':
                    self.state = b'bracket-escaped'
                    self.escBuf = []
                elif ch == b'O':
                    self.state = b'low-function-escaped'
                else:
                    self.state = b'data'
                    self._handleShortControlSequence(ch)
            elif self.state == b'bracket-escaped':
                if ch == b'O':
                    self.state = b'low-function-escaped'
                elif ch.isalpha() or ch == b'~':
                    self._handleControlSequence(b''.join(self.escBuf) + ch)
                    del self.escBuf
                    self.state = b'data'
                else:
                    self.escBuf.append(ch)
            elif self.state == b'low-function-escaped':
                self._handleLowFunctionControlSequence(ch)
                self.state = b'data'
            else:
                raise ValueError('Illegal state')

    def _handleShortControlSequence(self, ch):
        if False:
            for i in range(10):
                print('nop')
        self.terminalProtocol.keystrokeReceived(ch, self.ALT)

    def _handleControlSequence(self, buf):
        if False:
            for i in range(10):
                print('nop')
        buf = b'\x1b[' + buf
        f = getattr(self.controlSequenceParser, CST.get(buf[-1:], buf[-1:]).decode('ascii'), None)
        if f is None:
            self.unhandledControlSequence(buf)
        else:
            f(self, self.terminalProtocol, buf[:-1])

    def unhandledControlSequence(self, buf):
        if False:
            print('Hello World!')
        self.terminalProtocol.unhandledControlSequence(buf)

    def _handleLowFunctionControlSequence(self, ch):
        if False:
            for i in range(10):
                print('nop')
        functionKeys = {b'P': self.F1, b'Q': self.F2, b'R': self.F3, b'S': self.F4}
        keyID = functionKeys.get(ch)
        if keyID is not None:
            self.terminalProtocol.keystrokeReceived(keyID, None)
        else:
            self.terminalProtocol.unhandledControlSequence(b'\x1b[O' + ch)

    class ControlSequenceParser:

        def A(self, proto, handler, buf):
            if False:
                i = 10
                return i + 15
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.UP_ARROW, None)
            else:
                handler.unhandledControlSequence(buf + b'A')

        def B(self, proto, handler, buf):
            if False:
                print('Hello World!')
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.DOWN_ARROW, None)
            else:
                handler.unhandledControlSequence(buf + b'B')

        def C(self, proto, handler, buf):
            if False:
                for i in range(10):
                    print('nop')
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.RIGHT_ARROW, None)
            else:
                handler.unhandledControlSequence(buf + b'C')

        def D(self, proto, handler, buf):
            if False:
                return 10
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.LEFT_ARROW, None)
            else:
                handler.unhandledControlSequence(buf + b'D')

        def E(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.NUMPAD_MIDDLE, None)
            else:
                handler.unhandledControlSequence(buf + b'E')

        def F(self, proto, handler, buf):
            if False:
                print('Hello World!')
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.END, None)
            else:
                handler.unhandledControlSequence(buf + b'F')

        def H(self, proto, handler, buf):
            if False:
                for i in range(10):
                    print('nop')
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.HOME, None)
            else:
                handler.unhandledControlSequence(buf + b'H')

        def R(self, proto, handler, buf):
            if False:
                i = 10
                return i + 15
            if not proto._cursorReports:
                handler.unhandledControlSequence(buf + b'R')
            elif buf.startswith(b'\x1b['):
                report = buf[2:]
                parts = report.split(b';')
                if len(parts) != 2:
                    handler.unhandledControlSequence(buf + b'R')
                else:
                    (Pl, Pc) = parts
                    try:
                        (Pl, Pc) = (int(Pl), int(Pc))
                    except ValueError:
                        handler.unhandledControlSequence(buf + b'R')
                    else:
                        d = proto._cursorReports.pop(0)
                        d.callback((Pc - 1, Pl - 1))
            else:
                handler.unhandledControlSequence(buf + b'R')

        def Z(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            if buf == b'\x1b[':
                handler.keystrokeReceived(proto.TAB, proto.SHIFT)
            else:
                handler.unhandledControlSequence(buf + b'Z')

        def tilde(self, proto, handler, buf):
            if False:
                for i in range(10):
                    print('nop')
            map = {1: proto.HOME, 2: proto.INSERT, 3: proto.DELETE, 4: proto.END, 5: proto.PGUP, 6: proto.PGDN, 15: proto.F5, 17: proto.F6, 18: proto.F7, 19: proto.F8, 20: proto.F9, 21: proto.F10, 23: proto.F11, 24: proto.F12}
            if buf.startswith(b'\x1b['):
                ch = buf[2:]
                try:
                    v = int(ch)
                except ValueError:
                    handler.unhandledControlSequence(buf + b'~')
                else:
                    symbolic = map.get(v)
                    if symbolic is not None:
                        handler.keystrokeReceived(map[v], None)
                    else:
                        handler.unhandledControlSequence(buf + b'~')
            else:
                handler.unhandledControlSequence(buf + b'~')
    controlSequenceParser = ControlSequenceParser()

    def cursorUp(self, n=1):
        if False:
            return 10
        assert n >= 1
        self.cursorPos.y = max(self.cursorPos.y - n, 0)
        self.write(b'\x1b[%dA' % (n,))

    def cursorDown(self, n=1):
        if False:
            for i in range(10):
                print('nop')
        assert n >= 1
        self.cursorPos.y = min(self.cursorPos.y + n, self.termSize.y - 1)
        self.write(b'\x1b[%dB' % (n,))

    def cursorForward(self, n=1):
        if False:
            while True:
                i = 10
        assert n >= 1
        self.cursorPos.x = min(self.cursorPos.x + n, self.termSize.x - 1)
        self.write(b'\x1b[%dC' % (n,))

    def cursorBackward(self, n=1):
        if False:
            print('Hello World!')
        assert n >= 1
        self.cursorPos.x = max(self.cursorPos.x - n, 0)
        self.write(b'\x1b[%dD' % (n,))

    def cursorPosition(self, column, line):
        if False:
            while True:
                i = 10
        self.write(b'\x1b[%d;%dH' % (line + 1, column + 1))

    def cursorHome(self):
        if False:
            print('Hello World!')
        self.cursorPos.x = self.cursorPos.y = 0
        self.write(b'\x1b[H')

    def index(self):
        if False:
            i = 10
            return i + 15
        self.cursorPos.y = min(self.cursorPos.y + 1, self.termSize.y - 1)
        self.write(b'\x1bD')

    def reverseIndex(self):
        if False:
            while True:
                i = 10
        self.cursorPos.y = max(self.cursorPos.y - 1, 0)
        self.write(b'\x1bM')

    def nextLine(self):
        if False:
            i = 10
            return i + 15
        self.cursorPos.x = 0
        self.cursorPos.y = min(self.cursorPos.y + 1, self.termSize.y - 1)
        self.write(b'\n')

    def saveCursor(self):
        if False:
            while True:
                i = 10
        self._savedCursorPos = Vector(self.cursorPos.x, self.cursorPos.y)
        self.write(b'\x1b7')

    def restoreCursor(self):
        if False:
            i = 10
            return i + 15
        self.cursorPos = self._savedCursorPos
        del self._savedCursorPos
        self.write(b'\x1b8')

    def setModes(self, modes):
        if False:
            while True:
                i = 10
        modesBytes = b';'.join((b'%d' % (mode,) for mode in modes))
        self.write(b'\x1b[' + modesBytes + b'h')

    def setPrivateModes(self, modes):
        if False:
            for i in range(10):
                print('nop')
        modesBytes = b';'.join((b'%d' % (mode,) for mode in modes))
        self.write(b'\x1b[?' + modesBytes + b'h')

    def resetModes(self, modes):
        if False:
            for i in range(10):
                print('nop')
        modesBytes = b';'.join((b'%d' % (mode,) for mode in modes))
        self.write(b'\x1b[' + modesBytes + b'l')

    def resetPrivateModes(self, modes):
        if False:
            i = 10
            return i + 15
        modesBytes = b';'.join((b'%d' % (mode,) for mode in modes))
        self.write(b'\x1b[?' + modesBytes + b'l')

    def applicationKeypadMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x1b=')

    def numericKeypadMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x1b>')

    def selectCharacterSet(self, charSet, which):
        if False:
            return 10
        if which == G0:
            which = b'('
        elif which == G1:
            which = b')'
        else:
            raise ValueError("`which' argument to selectCharacterSet must be G0 or G1")
        if charSet == CS_UK:
            charSet = b'A'
        elif charSet == CS_US:
            charSet = b'B'
        elif charSet == CS_DRAWING:
            charSet = b'0'
        elif charSet == CS_ALTERNATE:
            charSet = b'1'
        elif charSet == CS_ALTERNATE_SPECIAL:
            charSet = b'2'
        else:
            raise ValueError("Invalid `charSet' argument to selectCharacterSet")
        self.write(b'\x1b' + which + charSet)

    def shiftIn(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x15')

    def shiftOut(self):
        if False:
            i = 10
            return i + 15
        self.write(b'\x14')

    def singleShift2(self):
        if False:
            i = 10
            return i + 15
        self.write(b'\x1bN')

    def singleShift3(self):
        if False:
            return 10
        self.write(b'\x1bO')

    def selectGraphicRendition(self, *attributes):
        if False:
            return 10
        attrs = []
        for a in attributes:
            attrs.append(networkString(a))
        self.write(b'\x1b[' + b';'.join(attrs) + b'm')

    def horizontalTabulationSet(self):
        if False:
            while True:
                i = 10
        self.write(b'\x1bH')

    def tabulationClear(self):
        if False:
            print('Hello World!')
        self.write(b'\x1b[q')

    def tabulationClearAll(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x1b[3q')

    def doubleHeightLine(self, top=True):
        if False:
            return 10
        if top:
            self.write(b'\x1b#3')
        else:
            self.write(b'\x1b#4')

    def singleWidthLine(self):
        if False:
            i = 10
            return i + 15
        self.write(b'\x1b#5')

    def doubleWidthLine(self):
        if False:
            while True:
                i = 10
        self.write(b'\x1b#6')

    def eraseToLineEnd(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x1b[K')

    def eraseToLineBeginning(self):
        if False:
            print('Hello World!')
        self.write(b'\x1b[1K')

    def eraseLine(self):
        if False:
            print('Hello World!')
        self.write(b'\x1b[2K')

    def eraseToDisplayEnd(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x1b[J')

    def eraseToDisplayBeginning(self):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'\x1b[1J')

    def eraseDisplay(self):
        if False:
            print('Hello World!')
        self.write(b'\x1b[2J')

    def deleteCharacter(self, n=1):
        if False:
            return 10
        self.write(b'\x1b[%dP' % (n,))

    def insertLine(self, n=1):
        if False:
            i = 10
            return i + 15
        self.write(b'\x1b[%dL' % (n,))

    def deleteLine(self, n=1):
        if False:
            print('Hello World!')
        self.write(b'\x1b[%dM' % (n,))

    def setScrollRegion(self, first=None, last=None):
        if False:
            i = 10
            return i + 15
        if first is not None:
            first = b'%d' % (first,)
        else:
            first = b''
        if last is not None:
            last = b'%d' % (last,)
        else:
            last = b''
        self.write(b'\x1b[%b;%br' % (first, last))

    def resetScrollRegion(self):
        if False:
            for i in range(10):
                print('nop')
        self.setScrollRegion()

    def reportCursorPosition(self):
        if False:
            return 10
        d = defer.Deferred()
        self._cursorReports.append(d)
        self.write(b'\x1b[6n')
        return d

    def reset(self):
        if False:
            return 10
        self.cursorPos.x = self.cursorPos.y = 0
        try:
            del self._savedCursorPos
        except AttributeError:
            pass
        self.write(b'\x1bc')

    def write(self, data):
        if False:
            while True:
                i = 10
        if data:
            if not isinstance(data, bytes):
                data = data.encode('utf-8')
            self.lastWrite = data
            self.transport.write(b'\r\n'.join(data.split(b'\n')))

    def writeSequence(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.write(b''.join(data))

    def loseConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.reset()
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            return 10
        if self.terminalProtocol is not None:
            try:
                self.terminalProtocol.connectionLost(reason)
            finally:
                self.terminalProtocol = None
for (name, const) in zip(_KEY_NAMES, FUNCTION_KEYS):
    setattr(ServerProtocol, name, const)

class ClientProtocol(protocol.Protocol):
    terminalFactory = None
    terminal = None
    state = b'data'
    _escBuf = None
    _shorts = {b'D': b'index', b'M': b'reverseIndex', b'E': b'nextLine', b'7': b'saveCursor', b'8': b'restoreCursor', b'=': b'applicationKeypadMode', b'>': b'numericKeypadMode', b'N': b'singleShift2', b'O': b'singleShift3', b'H': b'horizontalTabulationSet', b'c': b'reset'}
    _longs = {b'[': b'bracket-escape', b'(': b'select-g0', b')': b'select-g1', b'#': b'select-height-width'}
    _charsets = {b'A': CS_UK, b'B': CS_US, b'0': CS_DRAWING, b'1': CS_ALTERNATE, b'2': CS_ALTERNATE_SPECIAL}
    factory = None

    def __init__(self, terminalFactory=None, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param terminalFactory: A callable which will be invoked with\n        *a, **kw and should return an ITerminalTransport provider.\n        This will be invoked when this ClientProtocol establishes a\n        connection.\n\n        @param a: Any positional arguments to pass to terminalFactory.\n        @param kw: Any keyword arguments to pass to terminalFactory.\n        '
        if terminalFactory is not None:
            self.terminalFactory = terminalFactory
        self.terminalArgs = a
        self.terminalKwArgs = kw

    def connectionMade(self):
        if False:
            while True:
                i = 10
        if self.terminalFactory is not None:
            self.terminal = self.terminalFactory(*self.terminalArgs, **self.terminalKwArgs)
            self.terminal.factory = self.factory
            self.terminal.makeConnection(self)

    def connectionLost(self, reason):
        if False:
            return 10
        if self.terminal is not None:
            try:
                self.terminal.connectionLost(reason)
            finally:
                del self.terminal

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        '\n        Parse the given data from a terminal server, dispatching to event\n        handlers defined by C{self.terminal}.\n        '
        toWrite = []
        for b in iterbytes(data):
            if self.state == b'data':
                if b == b'\x1b':
                    if toWrite:
                        self.terminal.write(b''.join(toWrite))
                        del toWrite[:]
                    self.state = b'escaped'
                elif b == b'\x14':
                    if toWrite:
                        self.terminal.write(b''.join(toWrite))
                        del toWrite[:]
                    self.terminal.shiftOut()
                elif b == b'\x15':
                    if toWrite:
                        self.terminal.write(b''.join(toWrite))
                        del toWrite[:]
                    self.terminal.shiftIn()
                elif b == b'\x08':
                    if toWrite:
                        self.terminal.write(b''.join(toWrite))
                        del toWrite[:]
                    self.terminal.cursorBackward()
                else:
                    toWrite.append(b)
            elif self.state == b'escaped':
                fName = self._shorts.get(b)
                if fName is not None:
                    self.state = b'data'
                    getattr(self.terminal, fName.decode('ascii'))()
                else:
                    state = self._longs.get(b)
                    if state is not None:
                        self.state = state
                    else:
                        self.terminal.unhandledControlSequence(b'\x1b' + b)
                        self.state = b'data'
            elif self.state == b'bracket-escape':
                if self._escBuf is None:
                    self._escBuf = []
                if b.isalpha() or b == b'~':
                    self._handleControlSequence(b''.join(self._escBuf), b)
                    del self._escBuf
                    self.state = b'data'
                else:
                    self._escBuf.append(b)
            elif self.state == b'select-g0':
                self.terminal.selectCharacterSet(self._charsets.get(b, b), G0)
                self.state = b'data'
            elif self.state == b'select-g1':
                self.terminal.selectCharacterSet(self._charsets.get(b, b), G1)
                self.state = b'data'
            elif self.state == b'select-height-width':
                self._handleHeightWidth(b)
                self.state = b'data'
            else:
                raise ValueError('Illegal state')
        if toWrite:
            self.terminal.write(b''.join(toWrite))

    def _handleControlSequence(self, buf, terminal):
        if False:
            while True:
                i = 10
        f = getattr(self.controlSequenceParser, CST.get(terminal, terminal).decode('ascii'), None)
        if f is None:
            self.terminal.unhandledControlSequence(b'\x1b[' + buf + terminal)
        else:
            f(self, self.terminal, buf)

    class ControlSequenceParser:

        def _makeSimple(ch, fName):
            if False:
                print('Hello World!')
            n = 'cursor' + fName

            def simple(self, proto, handler, buf):
                if False:
                    print('Hello World!')
                if not buf:
                    getattr(handler, n)(1)
                else:
                    try:
                        m = int(buf)
                    except ValueError:
                        handler.unhandledControlSequence(b'\x1b[' + buf + ch)
                    else:
                        getattr(handler, n)(m)
            return simple
        for (ch, fName) in (('A', 'Up'), ('B', 'Down'), ('C', 'Forward'), ('D', 'Backward')):
            exec(ch + ' = _makeSimple(ch, fName)')
        del _makeSimple

        def h(self, proto, handler, buf):
            if False:
                return 10
            try:
                modes = [int(mode) for mode in buf.split(b';')]
            except ValueError:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'h')
            else:
                handler.setModes(modes)

        def l(self, proto, handler, buf):
            if False:
                print('Hello World!')
            try:
                modes = [int(mode) for mode in buf.split(b';')]
            except ValueError:
                handler.unhandledControlSequence(b'\x1b[' + buf + 'l')
            else:
                handler.resetModes(modes)

        def r(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            parts = buf.split(b';')
            if len(parts) == 1:
                handler.setScrollRegion(None, None)
            elif len(parts) == 2:
                try:
                    if parts[0]:
                        pt = int(parts[0])
                    else:
                        pt = None
                    if parts[1]:
                        pb = int(parts[1])
                    else:
                        pb = None
                except ValueError:
                    handler.unhandledControlSequence(b'\x1b[' + buf + b'r')
                else:
                    handler.setScrollRegion(pt, pb)
            else:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'r')

        def K(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            if not buf:
                handler.eraseToLineEnd()
            elif buf == b'1':
                handler.eraseToLineBeginning()
            elif buf == b'2':
                handler.eraseLine()
            else:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'K')

        def H(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            handler.cursorHome()

        def J(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            if not buf:
                handler.eraseToDisplayEnd()
            elif buf == b'1':
                handler.eraseToDisplayBeginning()
            elif buf == b'2':
                handler.eraseDisplay()
            else:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'J')

        def P(self, proto, handler, buf):
            if False:
                print('Hello World!')
            if not buf:
                handler.deleteCharacter(1)
            else:
                try:
                    n = int(buf)
                except ValueError:
                    handler.unhandledControlSequence(b'\x1b[' + buf + b'P')
                else:
                    handler.deleteCharacter(n)

        def L(self, proto, handler, buf):
            if False:
                return 10
            if not buf:
                handler.insertLine(1)
            else:
                try:
                    n = int(buf)
                except ValueError:
                    handler.unhandledControlSequence(b'\x1b[' + buf + b'L')
                else:
                    handler.insertLine(n)

        def M(self, proto, handler, buf):
            if False:
                print('Hello World!')
            if not buf:
                handler.deleteLine(1)
            else:
                try:
                    n = int(buf)
                except ValueError:
                    handler.unhandledControlSequence(b'\x1b[' + buf + b'M')
                else:
                    handler.deleteLine(n)

        def n(self, proto, handler, buf):
            if False:
                while True:
                    i = 10
            if buf == b'6':
                (x, y) = handler.reportCursorPosition()
                proto.transport.write(b'\x1b[%d;%dR' % (x + 1, y + 1))
            else:
                handler.unhandledControlSequence(b'\x1b[' + buf + b'n')

        def m(self, proto, handler, buf):
            if False:
                return 10
            if not buf:
                handler.selectGraphicRendition(NORMAL)
            else:
                attrs = []
                for a in buf.split(b';'):
                    try:
                        a = int(a)
                    except ValueError:
                        pass
                    attrs.append(a)
                handler.selectGraphicRendition(*attrs)
    controlSequenceParser = ControlSequenceParser()

    def _handleHeightWidth(self, b):
        if False:
            i = 10
            return i + 15
        if b == b'3':
            self.terminal.doubleHeightLine(True)
        elif b == b'4':
            self.terminal.doubleHeightLine(False)
        elif b == b'5':
            self.terminal.singleWidthLine()
        elif b == b'6':
            self.terminal.doubleWidthLine()
        else:
            self.terminal.unhandledControlSequence(b'\x1b#' + b)
__all__ = ['ITerminalProtocol', 'ITerminalTransport', 'modes', 'privateModes', 'FUNCTION_KEYS', 'CS_US', 'CS_UK', 'CS_DRAWING', 'CS_ALTERNATE', 'CS_ALTERNATE_SPECIAL', 'G0', 'G1', 'G2', 'G3', 'UNDERLINE', 'REVERSE_VIDEO', 'BLINK', 'BOLD', 'NORMAL', 'ServerProtocol', 'ClientProtocol']