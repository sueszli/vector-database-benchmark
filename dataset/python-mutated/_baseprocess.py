"""
Cross-platform process-related functionality used by different
L{IReactorProcess} implementations.
"""
from typing import Optional
from twisted.python.deprecate import getWarningMethod
from twisted.python.failure import Failure
from twisted.python.log import err
from twisted.python.reflect import qual
_missingProcessExited = 'Since Twisted 8.2, IProcessProtocol.processExited is required.  %s must implement it.'

class BaseProcess:
    pid: Optional[int] = None
    status: Optional[int] = None
    lostProcess = 0
    proto = None

    def __init__(self, protocol):
        if False:
            for i in range(10):
                print('nop')
        self.proto = protocol

    def _callProcessExited(self, reason):
        if False:
            return 10
        default = object()
        processExited = getattr(self.proto, 'processExited', default)
        if processExited is default:
            getWarningMethod()(_missingProcessExited % (qual(self.proto.__class__),), DeprecationWarning, stacklevel=0)
        else:
            try:
                processExited(Failure(reason))
            except BaseException:
                err(None, 'unexpected error in processExited')

    def processEnded(self, status):
        if False:
            print('Hello World!')
        '\n        This is called when the child terminates.\n        '
        self.status = status
        self.lostProcess += 1
        self.pid = None
        self._callProcessExited(self._getReason(status))
        self.maybeCallProcessEnded()

    def maybeCallProcessEnded(self):
        if False:
            i = 10
            return i + 15
        '\n        Call processEnded on protocol after final cleanup.\n        '
        if self.proto is not None:
            reason = self._getReason(self.status)
            proto = self.proto
            self.proto = None
            try:
                proto.processEnded(Failure(reason))
            except BaseException:
                err(None, 'unexpected error in processEnded')