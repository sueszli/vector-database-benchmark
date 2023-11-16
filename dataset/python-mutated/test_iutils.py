"""
Test running processes with the APIs in L{twisted.internet.utils}.
"""
import os
import signal
import stat
import sys
import warnings
from unittest import skipIf
from twisted.internet import error, interfaces, reactor, utils
from twisted.internet.defer import Deferred
from twisted.python.runtime import platform
from twisted.python.test.test_util import SuppressedWarningsTests
from twisted.trial.unittest import SynchronousTestCase, TestCase

class ProcessUtilsTests(TestCase):
    """
    Test running a process using L{getProcessOutput}, L{getProcessValue}, and
    L{getProcessOutputAndValue}.
    """
    if interfaces.IReactorProcess(reactor, None) is None:
        skip = "reactor doesn't implement IReactorProcess"
    output = None
    value = None
    exe = sys.executable

    def makeSourceFile(self, sourceLines):
        if False:
            while True:
                i = 10
        '\n        Write the given list of lines to a text file and return the absolute\n        path to it.\n        '
        script = self.mktemp()
        with open(script, 'wt') as scriptFile:
            scriptFile.write(os.linesep.join(sourceLines) + os.linesep)
        return os.path.abspath(script)

    def test_output(self):
        if False:
            return 10
        '\n        L{getProcessOutput} returns a L{Deferred} which fires with the complete\n        output of the process it runs after that process exits.\n        '
        scriptFile = self.makeSourceFile(['import sys', "for s in b'hello world\\n':", '    s = bytes([s])', '    sys.stdout.buffer.write(s)', '    sys.stdout.flush()'])
        d = utils.getProcessOutput(self.exe, ['-u', scriptFile])
        return d.addCallback(self.assertEqual, b'hello world\n')

    def test_outputWithErrorIgnored(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The L{Deferred} returned by L{getProcessOutput} is fired with an\n        L{IOError} L{Failure} if the child process writes to stderr.\n        '
        scriptFile = self.makeSourceFile(['import sys', 'sys.stderr.write("hello world\\n")'])
        d = utils.getProcessOutput(self.exe, ['-u', scriptFile])
        d = self.assertFailure(d, IOError)

        def cbFailed(err):
            if False:
                print('Hello World!')
            return self.assertFailure(err.processEnded, error.ProcessDone)
        d.addCallback(cbFailed)
        return d

    def test_outputWithErrorCollected(self):
        if False:
            print('Hello World!')
        "\n        If a C{True} value is supplied for the C{errortoo} parameter to\n        L{getProcessOutput}, the returned L{Deferred} fires with the child's\n        stderr output as well as its stdout output.\n        "
        scriptFile = self.makeSourceFile(['import sys', 'sys.stdout.write("foo")', 'sys.stdout.flush()', 'sys.stderr.write("foo")', 'sys.stderr.flush()'])
        d = utils.getProcessOutput(self.exe, ['-u', scriptFile], errortoo=True)
        return d.addCallback(self.assertEqual, b'foofoo')

    def test_value(self):
        if False:
            return 10
        '\n        The L{Deferred} returned by L{getProcessValue} is fired with the exit\n        status of the child process.\n        '
        scriptFile = self.makeSourceFile(['raise SystemExit(1)'])
        d = utils.getProcessValue(self.exe, ['-u', scriptFile])
        return d.addCallback(self.assertEqual, 1)

    def test_outputAndValue(self):
        if False:
            i = 10
            return i + 15
        "\n        The L{Deferred} returned by L{getProcessOutputAndValue} fires with a\n        three-tuple, the elements of which give the data written to the child's\n        stdout, the data written to the child's stderr, and the exit status of\n        the child.\n        "
        scriptFile = self.makeSourceFile(['import sys', "sys.stdout.buffer.write(b'hello world!\\n')", "sys.stderr.buffer.write(b'goodbye world!\\n')", 'sys.exit(1)'])

        def gotOutputAndValue(out_err_code):
            if False:
                print('Hello World!')
            (out, err, code) = out_err_code
            self.assertEqual(out, b'hello world!\n')
            self.assertEqual(err, b'goodbye world!\n')
            self.assertEqual(code, 1)
        d = utils.getProcessOutputAndValue(self.exe, ['-u', scriptFile])
        return d.addCallback(gotOutputAndValue)

    @skipIf(platform.isWindows(), "Windows doesn't have real signals.")
    def test_outputSignal(self):
        if False:
            i = 10
            return i + 15
        "\n        If the child process exits because of a signal, the L{Deferred}\n        returned by L{getProcessOutputAndValue} fires a L{Failure} of a tuple\n        containing the child's stdout, stderr, and the signal which caused\n        it to exit.\n        "
        scriptFile = self.makeSourceFile(['import sys, os, signal', "sys.stdout.write('stdout bytes\\n')", "sys.stderr.write('stderr bytes\\n')", 'sys.stdout.flush()', 'sys.stderr.flush()', 'os.kill(os.getpid(), signal.SIGKILL)'])

        def gotOutputAndValue(out_err_sig):
            if False:
                return 10
            (out, err, sig) = out_err_sig
            self.assertEqual(out, b'stdout bytes\n')
            self.assertEqual(err, b'stderr bytes\n')
            self.assertEqual(sig, signal.SIGKILL)
        d = utils.getProcessOutputAndValue(self.exe, ['-u', scriptFile])
        d = self.assertFailure(d, tuple)
        return d.addCallback(gotOutputAndValue)

    def _pathTest(self, utilFunc, check):
        if False:
            i = 10
            return i + 15
        dir = os.path.abspath(self.mktemp())
        os.makedirs(dir)
        scriptFile = self.makeSourceFile(['import os, sys', 'sys.stdout.write(os.getcwd())'])
        d = utilFunc(self.exe, ['-u', scriptFile], path=dir)
        d.addCallback(check, dir.encode(sys.getfilesystemencoding()))
        return d

    def test_getProcessOutputPath(self):
        if False:
            print('Hello World!')
        '\n        L{getProcessOutput} runs the given command with the working directory\n        given by the C{path} parameter.\n        '
        return self._pathTest(utils.getProcessOutput, self.assertEqual)

    def test_getProcessValuePath(self):
        if False:
            i = 10
            return i + 15
        '\n        L{getProcessValue} runs the given command with the working directory\n        given by the C{path} parameter.\n        '

        def check(result, ignored):
            if False:
                return 10
            self.assertEqual(result, 0)
        return self._pathTest(utils.getProcessValue, check)

    def test_getProcessOutputAndValuePath(self):
        if False:
            i = 10
            return i + 15
        '\n        L{getProcessOutputAndValue} runs the given command with the working\n        directory given by the C{path} parameter.\n        '

        def check(out_err_status, dir):
            if False:
                while True:
                    i = 10
            (out, err, status) = out_err_status
            self.assertEqual(out, dir)
            self.assertEqual(status, 0)
        return self._pathTest(utils.getProcessOutputAndValue, check)

    def _defaultPathTest(self, utilFunc, check):
        if False:
            return 10
        dir = os.path.abspath(self.mktemp())
        os.makedirs(dir)
        scriptFile = self.makeSourceFile(['import os, sys', 'cdir = os.getcwd()', 'sys.stdout.write(cdir)'])
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(dir)
        originalMode = stat.S_IMODE(os.stat('.').st_mode)
        os.chmod(dir, stat.S_IXUSR | stat.S_IRUSR)
        self.addCleanup(os.chmod, dir, originalMode)
        d = utilFunc(self.exe, ['-u', scriptFile])
        d.addCallback(check, dir.encode(sys.getfilesystemencoding()))
        return d

    def test_getProcessOutputDefaultPath(self):
        if False:
            i = 10
            return i + 15
        '\n        If no value is supplied for the C{path} parameter, L{getProcessOutput}\n        runs the given command in the same working directory as the parent\n        process and succeeds even if the current working directory is not\n        accessible.\n        '
        return self._defaultPathTest(utils.getProcessOutput, self.assertEqual)

    def test_getProcessValueDefaultPath(self):
        if False:
            while True:
                i = 10
        '\n        If no value is supplied for the C{path} parameter, L{getProcessValue}\n        runs the given command in the same working directory as the parent\n        process and succeeds even if the current working directory is not\n        accessible.\n        '

        def check(result, ignored):
            if False:
                return 10
            self.assertEqual(result, 0)
        return self._defaultPathTest(utils.getProcessValue, check)

    def test_getProcessOutputAndValueDefaultPath(self):
        if False:
            print('Hello World!')
        '\n        If no value is supplied for the C{path} parameter,\n        L{getProcessOutputAndValue} runs the given command in the same working\n        directory as the parent process and succeeds even if the current\n        working directory is not accessible.\n        '

        def check(out_err_status, dir):
            if False:
                print('Hello World!')
            (out, err, status) = out_err_status
            self.assertEqual(out, dir)
            self.assertEqual(status, 0)
        return self._defaultPathTest(utils.getProcessOutputAndValue, check)

    def test_get_processOutputAndValueStdin(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Standard input can be made available to the child process by passing\n        bytes for the `stdinBytes` parameter.\n        '
        scriptFile = self.makeSourceFile(['import sys', 'sys.stdout.write(sys.stdin.read())'])
        stdinBytes = b'These are the bytes to see.'
        d = utils.getProcessOutputAndValue(self.exe, ['-u', scriptFile], stdinBytes=stdinBytes)

        def gotOutputAndValue(out_err_code):
            if False:
                return 10
            (out, err, code) = out_err_code
            self.assertIn(stdinBytes, out)
            self.assertEqual(0, code)
        d.addCallback(gotOutputAndValue)
        return d

class SuppressWarningsTests(SynchronousTestCase):
    """
    Tests for L{utils.suppressWarnings}.
    """

    def test_suppressWarnings(self):
        if False:
            i = 10
            return i + 15
        '\n        L{utils.suppressWarnings} decorates a function so that the given\n        warnings are suppressed.\n        '
        result = []

        def showwarning(self, *a, **kw):
            if False:
                return 10
            result.append((a, kw))
        self.patch(warnings, 'showwarning', showwarning)

        def f(msg):
            if False:
                while True:
                    i = 10
            warnings.warn(msg)
        g = utils.suppressWarnings(f, (('ignore',), dict(message='This is message')))
        f('Sanity check message')
        self.assertEqual(len(result), 1)
        g('This is message')
        self.assertEqual(len(result), 1)
        g('Unignored message')
        self.assertEqual(len(result), 2)

class DeferredSuppressedWarningsTests(SuppressedWarningsTests):
    """
    Tests for L{utils.runWithWarningsSuppressed}, the version that supports
    Deferreds.
    """
    runWithWarningsSuppressed = staticmethod(utils.runWithWarningsSuppressed)

    def test_deferredCallback(self):
        if False:
            print('Hello World!')
        "\n        If the function called by L{utils.runWithWarningsSuppressed} returns a\n        C{Deferred}, the warning filters aren't removed until the Deferred\n        fires.\n        "
        filters = [(('ignore', '.*foo.*'), {}), (('ignore', '.*bar.*'), {})]
        result = Deferred()
        self.runWithWarningsSuppressed(filters, lambda : result)
        warnings.warn('ignore foo')
        result.callback(3)
        warnings.warn('ignore foo 2')
        self.assertEqual(['ignore foo 2'], [w['message'] for w in self.flushWarnings()])

    def test_deferredErrback(self):
        if False:
            print('Hello World!')
        "\n        If the function called by L{utils.runWithWarningsSuppressed} returns a\n        C{Deferred}, the warning filters aren't removed until the Deferred\n        fires with an errback.\n        "
        filters = [(('ignore', '.*foo.*'), {}), (('ignore', '.*bar.*'), {})]
        result = Deferred()
        d = self.runWithWarningsSuppressed(filters, lambda : result)
        warnings.warn('ignore foo')
        result.errback(ZeroDivisionError())
        d.addErrback(lambda f: f.trap(ZeroDivisionError))
        warnings.warn('ignore foo 2')
        self.assertEqual(['ignore foo 2'], [w['message'] for w in self.flushWarnings()])