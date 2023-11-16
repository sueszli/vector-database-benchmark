from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import log
from twisted.trial import unittest
from buildbot.test.util.warnings import assertProducesWarnings
from buildbot.util import lineboundaries
from buildbot.warnings import DeprecatedApiWarning

class LBF(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.callbacks = []
        with assertProducesWarnings(DeprecatedApiWarning, message_pattern='does not accept callback anymore'):
            self.lbf = lineboundaries.LineBoundaryFinder(self._callback)

    def _callback(self, wholeLines):
        if False:
            return 10
        self.assertEqual(wholeLines[-1], '\n', f'got {repr(wholeLines)}')
        self.callbacks.append(wholeLines)
        d = defer.Deferred()
        reactor.callLater(0, d.callback, None)
        return d

    def assertCallbacks(self, callbacks):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.callbacks, callbacks)
        self.callbacks = []

    @defer.inlineCallbacks
    def test_already_terminated(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.lbf.append('abcd\ndefg\n')
        self.assertCallbacks(['abcd\ndefg\n'])
        yield self.lbf.append('xyz\n')
        self.assertCallbacks(['xyz\n'])
        yield self.lbf.flush()
        self.assertCallbacks([])

    @defer.inlineCallbacks
    def test_partial_line(self):
        if False:
            while True:
                i = 10
        for c in 'hello\nworld':
            yield self.lbf.append(c)
        self.assertCallbacks(['hello\n'])
        yield self.lbf.flush()
        self.assertCallbacks(['world\n'])

    @defer.inlineCallbacks
    def test_empty_appends(self):
        if False:
            while True:
                i = 10
        yield self.lbf.append('hello ')
        yield self.lbf.append('')
        yield self.lbf.append('world\n')
        yield self.lbf.append('')
        self.assertCallbacks(['hello world\n'])

    @defer.inlineCallbacks
    def test_embedded_newlines(self):
        if False:
            i = 10
            return i + 15
        yield self.lbf.append('hello, ')
        self.assertCallbacks([])
        yield self.lbf.append('cruel\nworld')
        self.assertCallbacks(['hello, cruel\n'])
        yield self.lbf.flush()
        self.assertCallbacks(['world\n'])

    @defer.inlineCallbacks
    def test_windows_newlines_folded(self):
        if False:
            while True:
                i = 10
        "Windows' \\r\\n is treated as and converted to a newline"
        yield self.lbf.append('hello, ')
        self.assertCallbacks([])
        yield self.lbf.append('cruel\r\n\r\nworld')
        self.assertCallbacks(['hello, cruel\n\n'])
        yield self.lbf.flush()
        self.assertCallbacks(['world\n'])

    @defer.inlineCallbacks
    def test_bare_cr_folded(self):
        if False:
            print('Hello World!')
        'a bare \\r is treated as and converted to a newline'
        yield self.lbf.append('1%\r5%\r15%\r100%\nfinished')
        yield self.lbf.flush()
        self.assertCallbacks(['1%\n5%\n15%\n100%\n', 'finished\n'])

    @defer.inlineCallbacks
    def test_backspace_folded(self):
        if False:
            while True:
                i = 10
        'a lot of \\b is treated as and converted to a newline'
        yield self.lbf.append('1%\x08\x085%\x08\x0815%\x08\x08\x08100%\nfinished')
        yield self.lbf.flush()
        self.assertCallbacks(['1%\n5%\n15%\n100%\n', 'finished\n'])

    @defer.inlineCallbacks
    def test_mixed_consecutive_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        "mixing newline styles back-to-back doesn't collapse them"
        yield self.lbf.append('1\r\n\n\r')
        self.assertCallbacks(['1\n\n'])
        yield self.lbf.append('2\n\r\n')
        self.assertCallbacks(['\n2\n\n'])

    @defer.inlineCallbacks
    def test_split_newlines(self):
        if False:
            while True:
                i = 10
        'multi-character newlines, split across chunks, are converted'
        input = 'a\nb\r\nc\rd\n\re'
        for splitpoint in range(1, len(input) - 1):
            (a, b) = (input[:splitpoint], input[splitpoint:])
            yield self.lbf.append(a)
            yield self.lbf.append(b)
            yield self.lbf.flush()
            res = ''.join(self.callbacks)
            log.msg(f'feeding {repr(a)}, {repr(b)} gives {repr(res)}')
            self.assertEqual(res, 'a\nb\nc\nd\n\ne\n')
            self.callbacks = []

    @defer.inlineCallbacks
    def test_split_terminal_control(self):
        if False:
            print('Hello World!')
        'terminal control characters are converted'
        yield self.lbf.append('1234\x1b[u4321')
        yield self.lbf.flush()
        self.assertCallbacks(['1234\n', '4321\n'])
        yield self.lbf.append('1234\x1b[1;2H4321')
        yield self.lbf.flush()
        self.assertCallbacks(['1234\n', '4321\n'])
        yield self.lbf.append('1234\x1b[1;2f4321')
        yield self.lbf.flush()
        self.assertCallbacks(['1234\n', '4321\n'])

    @defer.inlineCallbacks
    def test_long_lines(self):
        if False:
            print('Hello World!')
        'long lines are split'
        for _ in range(4):
            yield self.lbf.append('12' * 1000)
        self.assertCallbacks(['12' * 2048 + '\n' + '12' * 952 + '\n'])

    @defer.inlineCallbacks
    def test_huge_lines(self):
        if False:
            print('Hello World!')
        'huge lines are split'
        yield self.lbf.append('12' * 32768)
        yield self.lbf.flush()
        self.assertCallbacks([('12' * 2048 + '\n') * 16])

    @defer.inlineCallbacks
    def test_empty_flush(self):
        if False:
            while True:
                i = 10
        yield self.lbf.flush()
        self.assertEqual(self.callbacks, [])

class LBFNoCallback(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.lbf = lineboundaries.LineBoundaryFinder()

    def test_already_terminated(self):
        if False:
            return 10
        res = self.lbf.append('abcd\ndefg\n')
        self.assertEqual(res, 'abcd\ndefg\n')
        res = self.lbf.append('xyz\n')
        self.assertEqual(res, 'xyz\n')
        res = self.lbf.flush()
        self.assertEqual(res, None)

    def test_partial_line(self):
        if False:
            print('Hello World!')
        res = self.lbf.append('hello\nworld')
        self.assertEqual(res, 'hello\n')
        res = self.lbf.flush()
        self.assertEqual(res, 'world\n')

    def test_empty_appends(self):
        if False:
            return 10
        res = self.lbf.append('hello ')
        self.assertEqual(res, None)
        res = self.lbf.append('')
        self.assertEqual(res, None)
        res = self.lbf.append('world\n')
        self.assertEqual(res, 'hello world\n')
        res = self.lbf.append('')
        self.assertEqual(res, None)

    def test_embedded_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.lbf.append('hello, ')
        self.assertEqual(res, None)
        res = self.lbf.append('cruel\nworld')
        self.assertEqual(res, 'hello, cruel\n')
        res = self.lbf.flush()
        self.assertEqual(res, 'world\n')

    def test_windows_newlines_folded(self):
        if False:
            for i in range(10):
                print('nop')
        "Windows' \\r\\n is treated as and converted to a newline"
        res = self.lbf.append('hello, ')
        self.assertEqual(res, None)
        res = self.lbf.append('cruel\r\n\r\nworld')
        self.assertEqual(res, 'hello, cruel\n\n')
        res = self.lbf.flush()
        self.assertEqual(res, 'world\n')

    def test_bare_cr_folded(self):
        if False:
            return 10
        'a bare \\r is treated as and converted to a newline'
        self.lbf.append('1%\r5%\r15%\r100%\nfinished')
        res = self.lbf.flush()
        self.assertEqual(res, 'finished\n')

    def test_backspace_folded(self):
        if False:
            return 10
        'a lot of \\b is treated as and converted to a newline'
        self.lbf.append('1%\x08\x085%\x08\x0815%\x08\x08\x08100%\nfinished')
        res = self.lbf.flush()
        self.assertEqual(res, 'finished\n')

    def test_mixed_consecutive_newlines(self):
        if False:
            while True:
                i = 10
        "mixing newline styles back-to-back doesn't collapse them"
        res = self.lbf.append('1\r\n\n\r')
        self.assertEqual(res, '1\n\n')
        res = self.lbf.append('2\n\r\n')
        self.assertEqual(res, '\n2\n\n')

    def test_split_newlines(self):
        if False:
            while True:
                i = 10
        'multi-character newlines, split across chunks, are converted'
        input = 'a\nb\r\nc\rd\n\re'
        result = []
        for splitpoint in range(1, len(input) - 1):
            (a, b) = (input[:splitpoint], input[splitpoint:])
            result.append(self.lbf.append(a))
            result.append(self.lbf.append(b))
            result.append(self.lbf.flush())
            result = [e for e in result if e is not None]
            res = ''.join(result)
            log.msg(f'feeding {repr(a)}, {repr(b)} gives {repr(res)}')
            self.assertEqual(res, 'a\nb\nc\nd\n\ne\n')
            result.clear()

    def test_split_terminal_control(self):
        if False:
            while True:
                i = 10
        'terminal control characters are converted'
        res = self.lbf.append('1234\x1b[u4321')
        self.assertEqual(res, '1234\n')
        res = self.lbf.flush()
        self.assertEqual(res, '4321\n')
        res = self.lbf.append('1234\x1b[1;2H4321')
        self.assertEqual(res, '1234\n')
        res = self.lbf.flush()
        self.assertEqual(res, '4321\n')
        res = self.lbf.append('1234\x1b[1;2f4321')
        self.assertEqual(res, '1234\n')
        res = self.lbf.flush()
        self.assertEqual(res, '4321\n')

    def test_long_lines(self):
        if False:
            return 10
        'long lines are split'
        res = []
        for _ in range(4):
            res.append(self.lbf.append('12' * 1000))
        res = [e for e in res if e is not None]
        res = ''.join(res)
        self.assertEqual(res, '12' * 2048 + '\n' + '12' * 952 + '\n')

    def test_huge_lines(self):
        if False:
            i = 10
            return i + 15
        'huge lines are split'
        res = []
        res.append(self.lbf.append('12' * 32768))
        res.append(self.lbf.flush())
        res = [e for e in res if e is not None]
        self.assertEqual(res, [('12' * 2048 + '\n') * 16])

    def test_empty_flush(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.lbf.flush()
        self.assertEqual(res, None)