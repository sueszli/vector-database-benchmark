"""
Test strerror
"""
import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase

class _MyWindowsException(OSError):
    """
    An exception type like L{ctypes.WinError}, but available on all platforms.
    """

class ErrorFormatingTests(TestCase):
    """
    Tests for C{_ErrorFormatter.formatError}.
    """
    probeErrorCode = ECONNABORTED
    probeMessage = 'correct message value'

    def test_strerrorFormatting(self):
        if False:
            i = 10
            return i + 15
        '\n        L{_ErrorFormatter.formatError} should use L{os.strerror} to format\n        error messages if it is constructed without any better mechanism.\n        '
        formatter = _ErrorFormatter(None, None, None)
        message = formatter.formatError(self.probeErrorCode)
        self.assertEqual(message, os.strerror(self.probeErrorCode))

    def test_emptyErrorTab(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_ErrorFormatter.formatError} should use L{os.strerror} to format\n        error messages if it is constructed with only an error tab which does\n        not contain the error code it is called with.\n        '
        error = 1
        self.assertNotEqual(self.probeErrorCode, error)
        formatter = _ErrorFormatter(None, None, {error: 'wrong message'})
        message = formatter.formatError(self.probeErrorCode)
        self.assertEqual(message, os.strerror(self.probeErrorCode))

    def test_errorTab(self):
        if False:
            print('Hello World!')
        '\n        L{_ErrorFormatter.formatError} should use C{errorTab} if it is supplied\n        and contains the requested error code.\n        '
        formatter = _ErrorFormatter(None, None, {self.probeErrorCode: self.probeMessage})
        message = formatter.formatError(self.probeErrorCode)
        self.assertEqual(message, self.probeMessage)

    def test_formatMessage(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_ErrorFormatter.formatError} should return the return value of\n        C{formatMessage} if it is supplied.\n        '
        formatCalls = []

        def formatMessage(errorCode):
            if False:
                for i in range(10):
                    print('nop')
            formatCalls.append(errorCode)
            return self.probeMessage
        formatter = _ErrorFormatter(None, formatMessage, {self.probeErrorCode: 'wrong message'})
        message = formatter.formatError(self.probeErrorCode)
        self.assertEqual(message, self.probeMessage)
        self.assertEqual(formatCalls, [self.probeErrorCode])

    def test_winError(self):
        if False:
            return 10
        '\n        L{_ErrorFormatter.formatError} should return the message argument from\n        the exception L{winError} returns, if L{winError} is supplied.\n        '
        winCalls = []

        def winError(errorCode):
            if False:
                for i in range(10):
                    print('nop')
            winCalls.append(errorCode)
            return _MyWindowsException(errorCode, self.probeMessage)
        formatter = _ErrorFormatter(winError, lambda error: 'formatMessage: wrong message', {self.probeErrorCode: 'errorTab: wrong message'})
        message = formatter.formatError(self.probeErrorCode)
        self.assertEqual(message, self.probeMessage)

    @skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
    def test_fromEnvironment(self):
        if False:
            print('Hello World!')
        '\n        L{_ErrorFormatter.fromEnvironment} should create an L{_ErrorFormatter}\n        instance with attributes populated from available modules.\n        '
        formatter = _ErrorFormatter.fromEnvironment()
        if formatter.winError is not None:
            from ctypes import WinError
            self.assertEqual(formatter.formatError(self.probeErrorCode), WinError(self.probeErrorCode).strerror)
            formatter.winError = None
        if formatter.formatMessage is not None:
            from win32api import FormatMessage
            self.assertEqual(formatter.formatError(self.probeErrorCode), FormatMessage(self.probeErrorCode))
            formatter.formatMessage = None
        if formatter.errorTab is not None:
            from socket import errorTab
            self.assertEqual(formatter.formatError(self.probeErrorCode), errorTab[self.probeErrorCode])

    @skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
    def test_correctLookups(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a known-good errno, make sure that formatMessage gives results\n        matching either C{socket.errorTab}, C{ctypes.WinError}, or\n        C{win32api.FormatMessage}.\n        '
        acceptable = [socket.errorTab[ECONNABORTED]]
        try:
            from ctypes import WinError
            acceptable.append(WinError(ECONNABORTED).strerror)
        except ImportError:
            pass
        try:
            from win32api import FormatMessage
            acceptable.append(FormatMessage(ECONNABORTED))
        except ImportError:
            pass
        self.assertIn(formatError(ECONNABORTED), acceptable)