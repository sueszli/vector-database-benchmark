"""
Tests for L{twisted.conch.manhole}.
"""
import sys
import traceback
from typing import Optional
ssh: Optional[bool] = None
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import _SSHMixin, _StdioMixin, _TelnetMixin, ssh, stdio
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest

def determineDefaultFunctionName():
    if False:
        while True:
            i = 10
    '\n    Return the string used by Python as the name for code objects which are\n    compiled from interactive input or at the top-level of modules.\n    '
    try:
        1 // 0
    except BaseException:
        return traceback.extract_stack()[-2][2]
defaultFunctionName = determineDefaultFunctionName()

class ManholeInterpreterTests(unittest.TestCase):
    """
    Tests for L{manhole.ManholeInterpreter}.
    """

    def test_resetBuffer(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ManholeInterpreter.resetBuffer} should empty the input buffer.\n        '
        interpreter = manhole.ManholeInterpreter(None)
        interpreter.buffer.extend(['1', '2'])
        interpreter.resetBuffer()
        self.assertFalse(interpreter.buffer)

class ManholeProtocolTests(unittest.TestCase):
    """
    Tests for L{manhole.Manhole}.
    """

    def test_interruptResetsInterpreterBuffer(self):
        if False:
            print('Hello World!')
        '\n        L{manhole.Manhole.handle_INT} should cause the interpreter input buffer\n        to be reset.\n        '
        transport = StringTransport()
        terminal = insults.ServerProtocol(manhole.Manhole)
        terminal.makeConnection(transport)
        protocol = terminal.terminalProtocol
        interpreter = protocol.interpreter
        interpreter.buffer.extend(['1', '2'])
        protocol.handle_INT()
        self.assertFalse(interpreter.buffer)

class WriterTests(unittest.TestCase):

    def test_Integer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Colorize an integer.\n        '
        manhole.lastColorizedLine('1')

    def test_DoubleQuoteString(self):
        if False:
            i = 10
            return i + 15
        '\n        Colorize an integer in double quotes.\n        '
        manhole.lastColorizedLine('"1"')

    def test_SingleQuoteString(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Colorize an integer in single quotes.\n        '
        manhole.lastColorizedLine("'1'")

    def test_TripleSingleQuotedString(self):
        if False:
            return 10
        '\n        Colorize an integer in triple quotes.\n        '
        manhole.lastColorizedLine("'''1'''")

    def test_TripleDoubleQuotedString(self):
        if False:
            print('Hello World!')
        '\n        Colorize an integer in triple and double quotes.\n        '
        manhole.lastColorizedLine('"""1"""')

    def test_FunctionDefinition(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Colorize a function definition.\n        '
        manhole.lastColorizedLine('def foo():')

    def test_ClassDefinition(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Colorize a class definition.\n        '
        manhole.lastColorizedLine('class foo:')

    def test_unicode(self):
        if False:
            return 10
        '\n        Colorize a Unicode string.\n        '
        res = manhole.lastColorizedLine('и')
        self.assertTrue(isinstance(res, bytes))

    def test_bytes(self):
        if False:
            while True:
                i = 10
        '\n        Colorize a UTF-8 byte string.\n        '
        res = manhole.lastColorizedLine(b'\xd0\xb8')
        self.assertTrue(isinstance(res, bytes))

    def test_identicalOutput(self):
        if False:
            print('Hello World!')
        '\n        The output of UTF-8 bytestrings and Unicode strings are identical.\n        '
        self.assertEqual(manhole.lastColorizedLine(b'\xd0\xb8'), manhole.lastColorizedLine('и'))

class ManholeLoopbackMixin:
    serverProtocol = manhole.ColoredManhole

    def test_SimpleExpression(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluate simple expression.\n        '
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b'1 + 1\ndone')

        def finished(ign):
            if False:
                i = 10
                return i + 15
            self._assertBuffer([b'>>> 1 + 1', b'2', b'>>> done'])
        return done.addCallback(finished)

    def test_TripleQuoteLineContinuation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluate line continuation in triple quotes.\n        '
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b"'''\n'''\ndone")

        def finished(ign):
            if False:
                return 10
            self._assertBuffer([b">>> '''", b"... '''", b"'\\n'", b'>>> done'])
        return done.addCallback(finished)

    def test_FunctionDefinition(self):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate function definition.\n        '
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b'def foo(bar):\n\tprint(bar)\n\nfoo(42)\ndone')

        def finished(ign):
            if False:
                return 10
            self._assertBuffer([b'>>> def foo(bar):', b'...     print(bar)', b'... ', b'>>> foo(42)', b'42', b'>>> done'])
        return done.addCallback(finished)

    def test_ClassDefinition(self):
        if False:
            return 10
        '\n        Evaluate class definition.\n        '
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b"class Foo:\n\tdef bar(self):\n\t\tprint('Hello, world!')\n\nFoo().bar()\ndone")

        def finished(ign):
            if False:
                print('Hello World!')
            self._assertBuffer([b'>>> class Foo:', b'...     def bar(self):', b"...         print('Hello, world!')", b'... ', b'>>> Foo().bar()', b'Hello, world!', b'>>> done'])
        return done.addCallback(finished)

    def test_Exception(self):
        if False:
            return 10
        '\n        Evaluate raising an exception.\n        '
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b"raise Exception('foo bar baz')\ndone")

        def finished(ign):
            if False:
                print('Hello World!')
            self._assertBuffer([b">>> raise Exception('foo bar baz')", b'Traceback (most recent call last):', b'  File "<console>", line 1, in ' + defaultFunctionName.encode('utf-8'), b'Exception: foo bar baz', b'>>> done'])
        done.addCallback(finished)
        return done

    def test_ExceptionWithCustomExcepthook(self):
        if False:
            return 10
        '\n        Raised exceptions are handled the same way even if L{sys.excepthook}\n        has been modified from its original value.\n        '
        self.patch(sys, 'excepthook', lambda *args: None)
        return self.test_Exception()

    def test_ControlC(self):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate interrupting with CTRL-C.\n        '
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b'cancelled line' + manhole.CTRL_C + b'done')

        def finished(ign):
            if False:
                for i in range(10):
                    print('nop')
            self._assertBuffer([b'>>> cancelled line', b'KeyboardInterrupt', b'>>> done'])
        return done.addCallback(finished)

    def test_interruptDuringContinuation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sending ^C to Manhole while in a state where more input is required to\n        complete a statement should discard the entire ongoing statement and\n        reset the input prompt to the non-continuation prompt.\n        '
        continuing = self.recvlineClient.expect(b'things')
        self._testwrite(b'(\nthings')

        def gotContinuation(ignored):
            if False:
                for i in range(10):
                    print('nop')
            self._assertBuffer([b'>>> (', b'... things'])
            interrupted = self.recvlineClient.expect(b'>>> ')
            self._testwrite(manhole.CTRL_C)
            return interrupted
        continuing.addCallback(gotContinuation)

        def gotInterruption(ignored):
            if False:
                i = 10
                return i + 15
            self._assertBuffer([b'>>> (', b'... things', b'KeyboardInterrupt', b'>>> '])
        continuing.addCallback(gotInterruption)
        return continuing

    def test_ControlBackslash(self):
        if False:
            print('Hello World!')
        '\n        Evaluate cancelling with CTRL-\\.\n        '
        self._testwrite(b'cancelled line')
        partialLine = self.recvlineClient.expect(b'cancelled line')

        def gotPartialLine(ign):
            if False:
                print('Hello World!')
            self._assertBuffer([b'>>> cancelled line'])
            self._testwrite(manhole.CTRL_BACKSLASH)
            d = self.recvlineClient.onDisconnection
            return self.assertFailure(d, error.ConnectionDone)

        def gotClearedLine(ign):
            if False:
                for i in range(10):
                    print('nop')
            self._assertBuffer([b''])
        return partialLine.addCallback(gotPartialLine).addCallback(gotClearedLine)

    @defer.inlineCallbacks
    def test_controlD(self):
        if False:
            while True:
                i = 10
        "\n        A CTRL+D in the middle of a line doesn't close a connection,\n        but at the beginning of a line it does.\n        "
        self._testwrite(b'1 + 1')
        yield self.recvlineClient.expect(b'\\+ 1')
        self._assertBuffer([b'>>> 1 + 1'])
        self._testwrite(manhole.CTRL_D + b' + 1')
        yield self.recvlineClient.expect(b'\\+ 1')
        self._assertBuffer([b'>>> 1 + 1 + 1'])
        self._testwrite(b'\n')
        yield self.recvlineClient.expect(b'3\n>>> ')
        self._testwrite(manhole.CTRL_D)
        d = self.recvlineClient.onDisconnection
        yield self.assertFailure(d, error.ConnectionDone)

    @defer.inlineCallbacks
    def test_ControlL(self):
        if False:
            print('Hello World!')
        "\n        CTRL+L is generally used as a redraw-screen command in terminal\n        applications.  Manhole doesn't currently respect this usage of it,\n        but it should at least do something reasonable in response to this\n        event (rather than, say, eating your face).\n        "
        self._testwrite(b'\n1 + 1')
        yield self.recvlineClient.expect(b'\\+ 1')
        self._assertBuffer([b'>>> ', b'>>> 1 + 1'])
        self._testwrite(manhole.CTRL_L + b' + 1')
        yield self.recvlineClient.expect(b'1 \\+ 1 \\+ 1')
        self._assertBuffer([b'>>> 1 + 1 + 1'])

    def test_controlA(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        CTRL-A can be used as HOME - returning cursor to beginning of\n        current line buffer.\n        '
        self._testwrite(b'rint "hello"' + b'\x01' + b'p')
        d = self.recvlineClient.expect(b'print "hello"')

        def cb(ignore):
            if False:
                for i in range(10):
                    print('nop')
            self._assertBuffer([b'>>> print "hello"'])
        return d.addCallback(cb)

    def test_controlE(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        CTRL-E can be used as END - setting cursor to end of current\n        line buffer.\n        '
        self._testwrite(b'rint "hello' + b'\x01' + b'p' + b'\x05' + b'"')
        d = self.recvlineClient.expect(b'print "hello"')

        def cb(ignore):
            if False:
                i = 10
                return i + 15
            self._assertBuffer([b'>>> print "hello"'])
        return d.addCallback(cb)

    @defer.inlineCallbacks
    def test_deferred(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a deferred is returned to the manhole REPL, it is displayed with\n        a sequence number, and when the deferred fires, the result is printed.\n        '
        self._testwrite(b'from twisted.internet import defer, reactor\nd = defer.Deferred()\nd\n')
        yield self.recvlineClient.expect(b'<Deferred #0>')
        self._testwrite(b"c = reactor.callLater(0.1, d.callback, 'Hi!')\n")
        yield self.recvlineClient.expect(b'>>> ')
        yield self.recvlineClient.expect(b"Deferred #0 called back: 'Hi!'\n>>> ")
        self._assertBuffer([b'>>> from twisted.internet import defer, reactor', b'>>> d = defer.Deferred()', b'>>> d', b'<Deferred #0>', b">>> c = reactor.callLater(0.1, d.callback, 'Hi!')", b"Deferred #0 called back: 'Hi!'", b'>>> '])

class ManholeLoopbackTelnetTests(_TelnetMixin, unittest.TestCase, ManholeLoopbackMixin):
    """
    Test manhole loopback over Telnet.
    """
    pass

class ManholeLoopbackSSHTests(_SSHMixin, unittest.TestCase, ManholeLoopbackMixin):
    """
    Test manhole loopback over SSH.
    """
    if ssh is None:
        skip = 'cryptography requirements missing'

class ManholeLoopbackStdioTests(_StdioMixin, unittest.TestCase, ManholeLoopbackMixin):
    """
    Test manhole loopback over standard IO.
    """
    if stdio is None:
        skip = 'Terminal requirements missing'
    else:
        serverProtocol = stdio.ConsoleManhole

class ManholeMainTests(unittest.TestCase):
    """
    Test the I{main} method from the I{manhole} module.
    """
    if stdio is None:
        skip = 'Terminal requirements missing'

    def test_mainClassNotFound(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Will raise an exception when called with an argument which is a\n        dotted patch which can not be imported..\n        '
        exception = self.assertRaises(ValueError, stdio.main, argv=['no-such-class'])
        self.assertEqual('Empty module name', exception.args[0])