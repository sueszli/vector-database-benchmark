"""
Select reactor
"""
import select
import sys
from errno import EBADF, EINTR
from time import sleep
from typing import Type
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
from twisted.python.runtime import platformType

def win32select(r, w, e, timeout=None):
    if False:
        for i in range(10):
            print('nop')
    'Win32 select wrapper.'
    if not (r or w):
        if timeout is None:
            timeout = 0.01
        else:
            timeout = min(timeout, 0.001)
        sleep(timeout)
        return ([], [], [])
    if timeout is None or timeout > 0.5:
        timeout = 0.5
    (r, w, e) = select.select(r, w, w, timeout)
    return (r, w + e, [])
if platformType == 'win32':
    _select = win32select
else:
    _select = select.select
try:
    from twisted.internet.win32eventreactor import _ThreadedWin32EventsMixin
except ImportError:
    _extraBase: Type[object] = object
else:
    _extraBase = _ThreadedWin32EventsMixin

@implementer(IReactorFDSet)
class SelectReactor(posixbase.PosixReactorBase, _extraBase):
    """
    A select() based reactor - runs on all POSIX platforms and on Win32.

    @ivar _reads: A set containing L{FileDescriptor} instances which will be
        checked for read events.

    @ivar _writes: A set containing L{FileDescriptor} instances which will be
        checked for writability.
    """

    def __init__(self):
        if False:
            return 10
        '\n        Initialize file descriptor tracking dictionaries and the base class.\n        '
        self._reads = set()
        self._writes = set()
        posixbase.PosixReactorBase.__init__(self)

    def _preenDescriptors(self):
        if False:
            print('Hello World!')
        log.msg('Malformed file descriptor found.  Preening lists.')
        readers = list(self._reads)
        writers = list(self._writes)
        self._reads.clear()
        self._writes.clear()
        for (selSet, selList) in ((self._reads, readers), (self._writes, writers)):
            for selectable in selList:
                try:
                    select.select([selectable], [selectable], [selectable], 0)
                except Exception as e:
                    log.msg('bad descriptor %s' % selectable)
                    self._disconnectSelectable(selectable, e, False)
                else:
                    selSet.add(selectable)

    def doSelect(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run one iteration of the I/O monitor loop.\n\n        This will run all selectables who had input or output readiness\n        waiting for them.\n        '
        try:
            (r, w, ignored) = _select(self._reads, self._writes, [], timeout)
        except ValueError:
            self._preenDescriptors()
            return
        except TypeError:
            log.err()
            self._preenDescriptors()
            return
        except OSError as se:
            if se.args[0] in (0, 2):
                if not self._reads and (not self._writes):
                    return
                else:
                    raise
            elif se.args[0] == EINTR:
                return
            elif se.args[0] == EBADF:
                self._preenDescriptors()
                return
            else:
                raise
        _drdw = self._doReadOrWrite
        _logrun = log.callWithLogger
        for (selectables, method, fdset) in ((r, 'doRead', self._reads), (w, 'doWrite', self._writes)):
            for selectable in selectables:
                if selectable not in fdset:
                    continue
                _logrun(selectable, _drdw, selectable, method)
    doIteration = doSelect

    def _doReadOrWrite(self, selectable, method):
        if False:
            i = 10
            return i + 15
        try:
            why = getattr(selectable, method)()
        except BaseException:
            why = sys.exc_info()[1]
            log.err()
        if why:
            self._disconnectSelectable(selectable, why, method == 'doRead')

    def addReader(self, reader):
        if False:
            return 10
        '\n        Add a FileDescriptor for notification of data available to read.\n        '
        self._reads.add(reader)

    def addWriter(self, writer):
        if False:
            print('Hello World!')
        '\n        Add a FileDescriptor for notification of data available to write.\n        '
        self._writes.add(writer)

    def removeReader(self, reader):
        if False:
            while True:
                i = 10
        '\n        Remove a Selectable for notification of data available to read.\n        '
        self._reads.discard(reader)

    def removeWriter(self, writer):
        if False:
            return 10
        '\n        Remove a Selectable for notification of data available to write.\n        '
        self._writes.discard(writer)

    def removeAll(self):
        if False:
            i = 10
            return i + 15
        return self._removeAll(self._reads, self._writes)

    def getReaders(self):
        if False:
            print('Hello World!')
        return list(self._reads)

    def getWriters(self):
        if False:
            i = 10
            return i + 15
        return list(self._writes)

def install():
    if False:
        i = 10
        return i + 15
    'Configure the twisted mainloop to be run using the select() reactor.'
    reactor = SelectReactor()
    from twisted.internet.main import installReactor
    installReactor(reactor)
__all__ = ['install']