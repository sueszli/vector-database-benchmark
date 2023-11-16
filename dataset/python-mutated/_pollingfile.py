"""
Implements a simple polling interface for file descriptors that don't work with
select() - this is pretty much only useful on Windows.
"""
from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
MIN_TIMEOUT = 1e-09
MAX_TIMEOUT = 0.1

class _PollableResource:
    active = True

    def activate(self):
        if False:
            i = 10
            return i + 15
        self.active = True

    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        self.active = False

class _PollingTimer:

    def __init__(self, reactor):
        if False:
            for i in range(10):
                print('nop')
        self.reactor = reactor
        self._resources = []
        self._pollTimer = None
        self._currentTimeout = MAX_TIMEOUT
        self._paused = False

    def _addPollableResource(self, res):
        if False:
            print('Hello World!')
        self._resources.append(res)
        self._checkPollingState()

    def _checkPollingState(self):
        if False:
            for i in range(10):
                print('nop')
        for resource in self._resources:
            if resource.active:
                self._startPolling()
                break
        else:
            self._stopPolling()

    def _startPolling(self):
        if False:
            print('Hello World!')
        if self._pollTimer is None:
            self._pollTimer = self._reschedule()

    def _stopPolling(self):
        if False:
            print('Hello World!')
        if self._pollTimer is not None:
            self._pollTimer.cancel()
            self._pollTimer = None

    def _pause(self):
        if False:
            for i in range(10):
                print('nop')
        self._paused = True

    def _unpause(self):
        if False:
            i = 10
            return i + 15
        self._paused = False
        self._checkPollingState()

    def _reschedule(self):
        if False:
            i = 10
            return i + 15
        if not self._paused:
            return self.reactor.callLater(self._currentTimeout, self._pollEvent)

    def _pollEvent(self):
        if False:
            return 10
        workUnits = 0.0
        anyActive = []
        for resource in self._resources:
            if resource.active:
                workUnits += resource.checkWork()
                if resource.active:
                    anyActive.append(resource)
        newTimeout = self._currentTimeout
        if workUnits:
            newTimeout = self._currentTimeout / (workUnits + 1.0)
            if newTimeout < MIN_TIMEOUT:
                newTimeout = MIN_TIMEOUT
        else:
            newTimeout = self._currentTimeout * 2.0
            if newTimeout > MAX_TIMEOUT:
                newTimeout = MAX_TIMEOUT
        self._currentTimeout = newTimeout
        if anyActive:
            self._pollTimer = self._reschedule()
import pywintypes
import win32api
import win32file
import win32pipe

@implementer(IPushProducer)
class _PollableReadPipe(_PollableResource):

    def __init__(self, pipe, receivedCallback, lostCallback):
        if False:
            return 10
        self.pipe = pipe
        self.receivedCallback = receivedCallback
        self.lostCallback = lostCallback

    def checkWork(self):
        if False:
            return 10
        finished = 0
        fullDataRead = []
        while 1:
            try:
                (buffer, bytesToRead, result) = win32pipe.PeekNamedPipe(self.pipe, 1)
                if not bytesToRead:
                    break
                (hr, data) = win32file.ReadFile(self.pipe, bytesToRead, None)
                fullDataRead.append(data)
            except win32api.error:
                finished = 1
                break
        dataBuf = b''.join(fullDataRead)
        if dataBuf:
            self.receivedCallback(dataBuf)
        if finished:
            self.cleanup()
        return len(dataBuf)

    def cleanup(self):
        if False:
            return 10
        self.deactivate()
        self.lostCallback()

    def close(self):
        if False:
            while True:
                i = 10
        try:
            win32api.CloseHandle(self.pipe)
        except pywintypes.error:
            pass

    def stopProducing(self):
        if False:
            return 10
        self.close()

    def pauseProducing(self):
        if False:
            print('Hello World!')
        self.deactivate()

    def resumeProducing(self):
        if False:
            for i in range(10):
                print('nop')
        self.activate()
FULL_BUFFER_SIZE = 64 * 1024

@implementer(IConsumer)
class _PollableWritePipe(_PollableResource):

    def __init__(self, writePipe, lostCallback):
        if False:
            i = 10
            return i + 15
        self.disconnecting = False
        self.producer = None
        self.producerPaused = False
        self.streamingProducer = 0
        self.outQueue = []
        self.writePipe = writePipe
        self.lostCallback = lostCallback
        try:
            win32pipe.SetNamedPipeHandleState(writePipe, win32pipe.PIPE_NOWAIT, None, None)
        except pywintypes.error:
            pass

    def close(self):
        if False:
            while True:
                i = 10
        self.disconnecting = True

    def bufferFull(self):
        if False:
            print('Hello World!')
        if self.producer is not None:
            self.producerPaused = True
            self.producer.pauseProducing()

    def bufferEmpty(self):
        if False:
            return 10
        if self.producer is not None and (not self.streamingProducer or self.producerPaused):
            self.producer.producerPaused = False
            self.producer.resumeProducing()
            return True
        return False

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        'Register to receive data from a producer.\n\n        This sets this selectable to be a consumer for a producer.  When this\n        selectable runs out of data on a write() call, it will ask the producer\n        to resumeProducing(). A producer should implement the IProducer\n        interface.\n\n        FileDescriptor provides some infrastructure for producer methods.\n        '
        if self.producer is not None:
            raise RuntimeError('Cannot register producer %s, because producer %s was never unregistered.' % (producer, self.producer))
        if not self.active:
            producer.stopProducing()
        else:
            self.producer = producer
            self.streamingProducer = streaming
            if not streaming:
                producer.resumeProducing()

    def unregisterProducer(self):
        if False:
            return 10
        'Stop consuming data from a producer, without disconnecting.'
        self.producer = None

    def writeConnectionLost(self):
        if False:
            i = 10
            return i + 15
        self.deactivate()
        try:
            win32api.CloseHandle(self.writePipe)
        except pywintypes.error:
            pass
        self.lostCallback()

    def writeSequence(self, seq):
        if False:
            print('Hello World!')
        '\n        Append a C{list} or C{tuple} of bytes to the output buffer.\n\n        @param seq: C{list} or C{tuple} of C{str} instances to be appended to\n            the output buffer.\n\n        @raise TypeError: If C{seq} contains C{unicode}.\n        '
        if str in map(type, seq):
            raise TypeError('Unicode not allowed in output buffer.')
        self.outQueue.extend(seq)

    def write(self, data):
        if False:
            return 10
        '\n        Append some bytes to the output buffer.\n\n        @param data: C{str} to be appended to the output buffer.\n        @type data: C{str}.\n\n        @raise TypeError: If C{data} is C{unicode} instead of C{str}.\n        '
        if isinstance(data, str):
            raise TypeError('Unicode not allowed in output buffer.')
        if self.disconnecting:
            return
        self.outQueue.append(data)
        if sum(map(len, self.outQueue)) > FULL_BUFFER_SIZE:
            self.bufferFull()

    def checkWork(self):
        if False:
            for i in range(10):
                print('nop')
        numBytesWritten = 0
        if not self.outQueue:
            if self.disconnecting:
                self.writeConnectionLost()
                return 0
            try:
                win32file.WriteFile(self.writePipe, b'', None)
            except pywintypes.error:
                self.writeConnectionLost()
                return numBytesWritten
        while self.outQueue:
            data = self.outQueue.pop(0)
            errCode = 0
            try:
                (errCode, nBytesWritten) = win32file.WriteFile(self.writePipe, data, None)
            except win32api.error:
                self.writeConnectionLost()
                break
            else:
                numBytesWritten += nBytesWritten
                if len(data) > nBytesWritten:
                    self.outQueue.insert(0, data[nBytesWritten:])
                    break
        else:
            resumed = self.bufferEmpty()
            if not resumed and self.disconnecting:
                self.writeConnectionLost()
        return numBytesWritten