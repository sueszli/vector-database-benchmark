"""Error formatting function for Windows.

The code is taken from twisted.python.win32 module.
"""
from __future__ import absolute_import
import os
__all__ = ['formatError']

class _ErrorFormatter(object):
    """
    Formatter for Windows error messages.

    @ivar winError: A callable which takes one integer error number argument
        and returns an L{exceptions.WindowsError} instance for that error (like
        L{ctypes.WinError}).

    @ivar formatMessage: A callable which takes one integer error number
        argument and returns a C{str} giving the message for that error (like
        L{win32api.FormatMessage}).

    @ivar errorTab: A mapping from integer error numbers to C{str} messages
        which correspond to those errors (like L{socket.errorTab}).
    """

    def __init__(self, WinError, FormatMessage, errorTab):
        if False:
            i = 10
            return i + 15
        self.winError = WinError
        self.formatMessage = FormatMessage
        self.errorTab = errorTab

    @classmethod
    def fromEnvironment(cls):
        if False:
            return 10
        '\n        Get as many of the platform-specific error translation objects as\n        possible and return an instance of C{cls} created with them.\n        '
        try:
            from ctypes import WinError
        except ImportError:
            WinError = None
        try:
            from win32api import FormatMessage
        except ImportError:
            FormatMessage = None
        try:
            from socket import errorTab
        except ImportError:
            errorTab = None
        return cls(WinError, FormatMessage, errorTab)

    def formatError(self, errorcode):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the string associated with a Windows error message, such as the\n        ones found in socket.error.\n\n        Attempts direct lookup against the win32 API via ctypes and then\n        pywin32 if available), then in the error table in the socket module,\n        then finally defaulting to C{os.strerror}.\n\n        @param errorcode: the Windows error code\n        @type errorcode: C{int}\n\n        @return: The error message string\n        @rtype: C{str}\n        '
        if self.winError is not None:
            return str(self.winError(errorcode))
        if self.formatMessage is not None:
            return self.formatMessage(errorcode)
        if self.errorTab is not None:
            result = self.errorTab.get(errorcode)
            if result is not None:
                return result
        return os.strerror(errorcode)
formatError = _ErrorFormatter.fromEnvironment().formatError