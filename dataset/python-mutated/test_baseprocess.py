"""
Tests for L{twisted.internet._baseprocess} which implements process-related
functionality that is useful in all platforms supporting L{IReactorProcess}.
"""
from twisted.internet._baseprocess import BaseProcess
from twisted.python.deprecate import getWarningMethod, setWarningMethod
from twisted.trial.unittest import TestCase

class BaseProcessTests(TestCase):
    """
    Tests for L{BaseProcess}, a parent class for other classes which represent
    processes which implements functionality common to many different process
    implementations.
    """

    def test_callProcessExited(self):
        if False:
            i = 10
            return i + 15
        '\n        L{BaseProcess._callProcessExited} calls the C{processExited} method of\n        its C{proto} attribute and passes it a L{Failure} wrapping the given\n        exception.\n        '

        class FakeProto:
            reason = None

            def processExited(self, reason):
                if False:
                    while True:
                        i = 10
                self.reason = reason
        reason = RuntimeError('fake reason')
        process = BaseProcess(FakeProto())
        process._callProcessExited(reason)
        process.proto.reason.trap(RuntimeError)
        self.assertIs(reason, process.proto.reason.value)

    def test_callProcessExitedMissing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{BaseProcess._callProcessExited} emits a L{DeprecationWarning} if the\n        object referred to by its C{proto} attribute has no C{processExited}\n        method.\n        '

        class FakeProto:
            pass
        reason = object()
        process = BaseProcess(FakeProto())
        self.addCleanup(setWarningMethod, getWarningMethod())
        warnings = []

        def collect(message, category, stacklevel):
            if False:
                i = 10
                return i + 15
            warnings.append((message, category, stacklevel))
        setWarningMethod(collect)
        process._callProcessExited(reason)
        [(message, category, stacklevel)] = warnings
        self.assertEqual(message, 'Since Twisted 8.2, IProcessProtocol.processExited is required.  %s.%s must implement it.' % (FakeProto.__module__, FakeProto.__name__))
        self.assertIs(category, DeprecationWarning)
        self.assertEqual(stacklevel, 0)