"""
Basic line editing support.

@author: Jp Calderone
"""
import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
_counters: Dict[str, int] = {}

class Logging:
    """
    Wrapper which logs attribute lookups.

    This was useful in debugging something, I guess.  I forget what.
    It can probably be deleted or moved somewhere more appropriate.
    Nothing special going on here, really.
    """

    def __init__(self, original):
        if False:
            for i in range(10):
                print('nop')
        self.original = original
        key = reflect.qual(original.__class__)
        count = _counters.get(key, 0)
        _counters[key] = count + 1
        self._logFile = open(key + '-' + str(count), 'w')

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return str(super().__getattribute__('original'))

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return repr(super().__getattribute__('original'))

    def __getattribute__(self, name):
        if False:
            i = 10
            return i + 15
        original = super().__getattribute__('original')
        logFile = super().__getattribute__('_logFile')
        logFile.write(name + '\n')
        return getattr(original, name)

@implementer(insults.ITerminalTransport)
class TransportSequence:
    """
    An L{ITerminalTransport} implementation which forwards calls to
    one or more other L{ITerminalTransport}s.

    This is a cheap way for servers to keep track of the state they
    expect the client to see, since all terminal manipulations can be
    send to the real client and to a terminal emulator that lives in
    the server process.
    """
    for keyID in (b'UP_ARROW', b'DOWN_ARROW', b'RIGHT_ARROW', b'LEFT_ARROW', b'HOME', b'INSERT', b'DELETE', b'END', b'PGUP', b'PGDN', b'F1', b'F2', b'F3', b'F4', b'F5', b'F6', b'F7', b'F8', b'F9', b'F10', b'F11', b'F12'):
        execBytes = keyID + b' = object()'
        execStr = execBytes.decode('ascii')
        exec(execStr)
    TAB = b'\t'
    BACKSPACE = b'\x7f'

    def __init__(self, *transports):
        if False:
            while True:
                i = 10
        assert transports, 'Cannot construct a TransportSequence with no transports'
        self.transports = transports
    for method in insults.ITerminalTransport:
        exec('def %s(self, *a, **kw):\n    for tpt in self.transports:\n        result = tpt.%s(*a, **kw)\n    return result\n' % (method, method))

    def getHost(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.getHost')

    def getPeer(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.getPeer')

    def loseConnection(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.loseConnection')

    def write(self, data):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.write')

    def writeSequence(self, data):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.writeSequence')

    def cursorUp(self, n=1):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Unimplemented: TransportSequence.cursorUp')

    def cursorDown(self, n=1):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Unimplemented: TransportSequence.cursorDown')

    def cursorForward(self, n=1):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Unimplemented: TransportSequence.cursorForward')

    def cursorBackward(self, n=1):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.cursorBackward')

    def cursorPosition(self, column, line):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.cursorPosition')

    def cursorHome(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.cursorHome')

    def index(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.index')

    def reverseIndex(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.reverseIndex')

    def nextLine(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.nextLine')

    def saveCursor(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.saveCursor')

    def restoreCursor(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.restoreCursor')

    def setModes(self, modes):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.setModes')

    def resetModes(self, mode):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.resetModes')

    def setPrivateModes(self, modes):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.setPrivateModes')

    def resetPrivateModes(self, modes):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.resetPrivateModes')

    def applicationKeypadMode(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.applicationKeypadMode')

    def numericKeypadMode(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.numericKeypadMode')

    def selectCharacterSet(self, charSet, which):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.selectCharacterSet')

    def shiftIn(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.shiftIn')

    def shiftOut(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.shiftOut')

    def singleShift2(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.singleShift2')

    def singleShift3(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.singleShift3')

    def selectGraphicRendition(self, *attributes):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.selectGraphicRendition')

    def horizontalTabulationSet(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.horizontalTabulationSet')

    def tabulationClear(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.tabulationClear')

    def tabulationClearAll(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.tabulationClearAll')

    def doubleHeightLine(self, top=True):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.doubleHeightLine')

    def singleWidthLine(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.singleWidthLine')

    def doubleWidthLine(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.doubleWidthLine')

    def eraseToLineEnd(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToLineEnd')

    def eraseToLineBeginning(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToLineBeginning')

    def eraseLine(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Unimplemented: TransportSequence.eraseLine')

    def eraseToDisplayEnd(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToDisplayEnd')

    def eraseToDisplayBeginning(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToDisplayBeginning')

    def eraseDisplay(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TransportSequence.eraseDisplay')

    def deleteCharacter(self, n=1):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.deleteCharacter')

    def insertLine(self, n=1):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.insertLine')

    def deleteLine(self, n=1):
        if False:
            print('Hello World!')
        raise NotImplementedError('Unimplemented: TransportSequence.deleteLine')

    def reportCursorPosition(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.reportCursorPosition')

    def reset(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: TransportSequence.reset')

    def unhandledControlSequence(self, seq):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: TransportSequence.unhandledControlSequence')

class LocalTerminalBufferMixin:
    """
    A mixin for RecvLine subclasses which records the state of the terminal.

    This is accomplished by performing all L{ITerminalTransport} operations on both
    the transport passed to makeConnection and an instance of helper.TerminalBuffer.

    @ivar terminalCopy: A L{helper.TerminalBuffer} instance which efforts
    will be made to keep up to date with the actual terminal
    associated with this protocol instance.
    """

    def makeConnection(self, transport):
        if False:
            print('Hello World!')
        self.terminalCopy = helper.TerminalBuffer()
        self.terminalCopy.connectionMade()
        return super().makeConnection(TransportSequence(transport, self.terminalCopy))

    def __str__(self) -> str:
        if False:
            return 10
        return str(self.terminalCopy)

class RecvLine(insults.TerminalProtocol):
    """
    L{TerminalProtocol} which adds line editing features.

    Clients will be prompted for lines of input with all the usual
    features: character echoing, left and right arrow support for
    moving the cursor to different areas of the line buffer, backspace
    and delete for removing characters, and insert for toggling
    between typeover and insert mode.  Tabs will be expanded to enough
    spaces to move the cursor to the next tabstop (every four
    characters by default).  Enter causes the line buffer to be
    cleared and the line to be passed to the lineReceived() method
    which, by default, does nothing.  Subclasses are responsible for
    redrawing the input prompt (this will probably change).
    """
    width = 80
    height = 24
    TABSTOP = 4
    ps = (b'>>> ', b'... ')
    pn = 0
    _printableChars = string.printable.encode('ascii')
    _log = Logger()

    def connectionMade(self):
        if False:
            while True:
                i = 10
        self.lineBuffer = []
        self.lineBufferIndex = 0
        t = self.terminal
        self.keyHandlers = {t.LEFT_ARROW: self.handle_LEFT, t.RIGHT_ARROW: self.handle_RIGHT, t.TAB: self.handle_TAB, b'\r': self.handle_RETURN, b'\n': self.handle_RETURN, t.BACKSPACE: self.handle_BACKSPACE, t.DELETE: self.handle_DELETE, t.INSERT: self.handle_INSERT, t.HOME: self.handle_HOME, t.END: self.handle_END}
        self.initializeScreen()

    def initializeScreen(self):
        if False:
            i = 10
            return i + 15
        self.terminal.reset()
        self.terminal.write(self.ps[self.pn])
        self.setInsertMode()

    def currentLineBuffer(self):
        if False:
            i = 10
            return i + 15
        s = b''.join(self.lineBuffer)
        return (s[:self.lineBufferIndex], s[self.lineBufferIndex:])

    def setInsertMode(self):
        if False:
            return 10
        self.mode = 'insert'
        self.terminal.setModes([insults.modes.IRM])

    def setTypeoverMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.mode = 'typeover'
        self.terminal.resetModes([insults.modes.IRM])

    def drawInputLine(self):
        if False:
            return 10
        '\n        Write a line containing the current input prompt and the current line\n        buffer at the current cursor position.\n        '
        self.terminal.write(self.ps[self.pn] + b''.join(self.lineBuffer))

    def terminalSize(self, width, height):
        if False:
            for i in range(10):
                print('nop')
        self.terminal.eraseDisplay()
        self.terminal.cursorHome()
        self.width = width
        self.height = height
        self.drawInputLine()

    def unhandledControlSequence(self, seq):
        if False:
            for i in range(10):
                print('nop')
        pass

    def keystrokeReceived(self, keyID, modifier):
        if False:
            return 10
        m = self.keyHandlers.get(keyID)
        if m is not None:
            m()
        elif keyID in self._printableChars:
            self.characterReceived(keyID, False)
        else:
            self._log.warn('Received unhandled keyID: {keyID!r}', keyID=keyID)

    def characterReceived(self, ch, moreCharactersComing):
        if False:
            for i in range(10):
                print('nop')
        if self.mode == 'insert':
            self.lineBuffer.insert(self.lineBufferIndex, ch)
        else:
            self.lineBuffer[self.lineBufferIndex:self.lineBufferIndex + 1] = [ch]
        self.lineBufferIndex += 1
        self.terminal.write(ch)

    def handle_TAB(self):
        if False:
            i = 10
            return i + 15
        n = self.TABSTOP - len(self.lineBuffer) % self.TABSTOP
        self.terminal.cursorForward(n)
        self.lineBufferIndex += n
        self.lineBuffer.extend(iterbytes(b' ' * n))

    def handle_LEFT(self):
        if False:
            print('Hello World!')
        if self.lineBufferIndex > 0:
            self.lineBufferIndex -= 1
            self.terminal.cursorBackward()

    def handle_RIGHT(self):
        if False:
            while True:
                i = 10
        if self.lineBufferIndex < len(self.lineBuffer):
            self.lineBufferIndex += 1
            self.terminal.cursorForward()

    def handle_HOME(self):
        if False:
            return 10
        if self.lineBufferIndex:
            self.terminal.cursorBackward(self.lineBufferIndex)
            self.lineBufferIndex = 0

    def handle_END(self):
        if False:
            while True:
                i = 10
        offset = len(self.lineBuffer) - self.lineBufferIndex
        if offset:
            self.terminal.cursorForward(offset)
            self.lineBufferIndex = len(self.lineBuffer)

    def handle_BACKSPACE(self):
        if False:
            i = 10
            return i + 15
        if self.lineBufferIndex > 0:
            self.lineBufferIndex -= 1
            del self.lineBuffer[self.lineBufferIndex]
            self.terminal.cursorBackward()
            self.terminal.deleteCharacter()

    def handle_DELETE(self):
        if False:
            i = 10
            return i + 15
        if self.lineBufferIndex < len(self.lineBuffer):
            del self.lineBuffer[self.lineBufferIndex]
            self.terminal.deleteCharacter()

    def handle_RETURN(self):
        if False:
            print('Hello World!')
        line = b''.join(self.lineBuffer)
        self.lineBuffer = []
        self.lineBufferIndex = 0
        self.terminal.nextLine()
        self.lineReceived(line)

    def handle_INSERT(self):
        if False:
            return 10
        assert self.mode in ('typeover', 'insert')
        if self.mode == 'typeover':
            self.setInsertMode()
        else:
            self.setTypeoverMode()

    def lineReceived(self, line):
        if False:
            print('Hello World!')
        pass

class HistoricRecvLine(RecvLine):
    """
    L{TerminalProtocol} which adds both basic line-editing features and input history.

    Everything supported by L{RecvLine} is also supported by this class.  In addition, the
    up and down arrows traverse the input history.  Each received line is automatically
    added to the end of the input history.
    """

    def connectionMade(self):
        if False:
            while True:
                i = 10
        RecvLine.connectionMade(self)
        self.historyLines = []
        self.historyPosition = 0
        t = self.terminal
        self.keyHandlers.update({t.UP_ARROW: self.handle_UP, t.DOWN_ARROW: self.handle_DOWN})

    def currentHistoryBuffer(self):
        if False:
            return 10
        b = tuple(self.historyLines)
        return (b[:self.historyPosition], b[self.historyPosition:])

    def _deliverBuffer(self, buf):
        if False:
            return 10
        if buf:
            for ch in iterbytes(buf[:-1]):
                self.characterReceived(ch, True)
            self.characterReceived(buf[-1:], False)

    def handle_UP(self):
        if False:
            for i in range(10):
                print('nop')
        if self.lineBuffer and self.historyPosition == len(self.historyLines):
            self.historyLines.append(b''.join(self.lineBuffer))
        if self.historyPosition > 0:
            self.handle_HOME()
            self.terminal.eraseToLineEnd()
            self.historyPosition -= 1
            self.lineBuffer = []
            self._deliverBuffer(self.historyLines[self.historyPosition])

    def handle_DOWN(self):
        if False:
            for i in range(10):
                print('nop')
        if self.historyPosition < len(self.historyLines) - 1:
            self.handle_HOME()
            self.terminal.eraseToLineEnd()
            self.historyPosition += 1
            self.lineBuffer = []
            self._deliverBuffer(self.historyLines[self.historyPosition])
        else:
            self.handle_HOME()
            self.terminal.eraseToLineEnd()
            self.historyPosition = len(self.historyLines)
            self.lineBuffer = []
            self.lineBufferIndex = 0

    def handle_RETURN(self):
        if False:
            print('Hello World!')
        if self.lineBuffer:
            self.historyLines.append(b''.join(self.lineBuffer))
        self.historyPosition = len(self.historyLines)
        return RecvLine.handle_RETURN(self)