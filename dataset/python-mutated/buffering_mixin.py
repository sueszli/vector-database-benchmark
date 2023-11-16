"""
Benchmarks comparing the write performance of a "normal" Protocol instance
and an instance of a Protocol class which has had L{twisted.conch.mixin}'s
L{BufferingMixin<twisted.conch.mixin.BufferingMixin>} mixed in to perform
Nagle-like write coalescing.
"""
from pprint import pprint
from sys import stdout
from time import time
from twisted.conch.mixin import BufferingMixin
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.protocol import ClientCreator, Protocol, ServerFactory
from twisted.python.log import startLogging
from twisted.python.usage import Options

class BufferingBenchmark(Options):
    """
    Options for configuring the execution parameters of a benchmark run.
    """
    optParameters = [('scale', 's', '1', 'Work multiplier (bigger takes longer, might resist noise better)')]

    def postOptions(self):
        if False:
            i = 10
            return i + 15
        self['scale'] = int(self['scale'])

class ServerProtocol(Protocol):
    """
    A silent protocol which only waits for a particular amount of input and
    then fires a Deferred.
    """

    def __init__(self, expected, finished):
        if False:
            i = 10
            return i + 15
        self.expected = expected
        self.finished = finished

    def dataReceived(self, bytes):
        if False:
            for i in range(10):
                print('nop')
        self.expected -= len(bytes)
        if self.expected == 0:
            (finished, self.finished) = (self.finished, None)
            finished.callback(None)

class BufferingProtocol(Protocol, BufferingMixin):
    """
    A protocol which uses the buffering mixin to provide a write method.
    """

class UnbufferingProtocol(Protocol):
    """
    A protocol which provides a naive write method which simply passes through
    to the transport.
    """

    def connectionMade(self):
        if False:
            while True:
                i = 10
        "\n        Bind write to the transport's write method and flush to a no-op\n        function in order to provide the same API as is provided by\n        BufferingProtocol.\n        "
        self.write = self.transport.write
        self.flush = lambda : None

def _write(proto, byteCount):
    if False:
        print('Hello World!')
    write = proto.write
    flush = proto.flush
    for i in range(byteCount):
        write('x')
    flush()

def _benchmark(byteCount, clientProtocol):
    if False:
        return 10
    result = {}
    finished = Deferred()

    def cbFinished(ignored):
        if False:
            return 10
        result['disconnected'] = time()
        result['duration'] = result['disconnected'] - result['connected']
        return result
    finished.addCallback(cbFinished)
    f = ServerFactory()
    f.protocol = lambda : ServerProtocol(byteCount, finished)
    server = reactor.listenTCP(0, f)
    f2 = ClientCreator(reactor, clientProtocol)
    proto = f2.connectTCP('127.0.0.1', server.getHost().port)

    def connected(proto):
        if False:
            for i in range(10):
                print('nop')
        result['connected'] = time()
        return proto
    proto.addCallback(connected)
    proto.addCallback(_write, byteCount)
    return finished

def _benchmarkBuffered(byteCount):
    if False:
        i = 10
        return i + 15
    return _benchmark(byteCount, BufferingProtocol)

def _benchmarkUnbuffered(byteCount):
    if False:
        i = 10
        return i + 15
    return _benchmark(byteCount, UnbufferingProtocol)

def benchmark(scale=1):
    if False:
        return 10
    "\n    Benchmark and return information regarding the relative performance of a\n    protocol which does not use the buffering mixin and a protocol which\n    does.\n\n    @type scale: C{int}\n    @param scale: A multiplier to the amount of work to perform\n\n    @return: A Deferred which will fire with a dictionary mapping each of\n    the two unicode strings C{u'buffered'} and C{u'unbuffered'} to\n    dictionaries describing the performance of a protocol of each type.\n    These value dictionaries will map the unicode strings C{u'connected'}\n    and C{u'disconnected'} to the times at which each of those events\n    occurred and C{u'duration'} two the difference between these two values.\n    "
    overallResult = {}
    byteCount = 1024
    bufferedDeferred = _benchmarkBuffered(byteCount * scale)

    def didBuffered(bufferedResult):
        if False:
            for i in range(10):
                print('nop')
        overallResult['buffered'] = bufferedResult
        unbufferedDeferred = _benchmarkUnbuffered(byteCount * scale)

        def didUnbuffered(unbufferedResult):
            if False:
                for i in range(10):
                    print('nop')
            overallResult['unbuffered'] = unbufferedResult
            return overallResult
        unbufferedDeferred.addCallback(didUnbuffered)
        return unbufferedDeferred
    bufferedDeferred.addCallback(didBuffered)
    return bufferedDeferred

def main(args=None):
    if False:
        i = 10
        return i + 15
    '\n    Perform a single benchmark run, starting and stopping the reactor and\n    logging system as necessary.\n    '
    startLogging(stdout)
    options = BufferingBenchmark()
    options.parseOptions(args)
    d = benchmark(options['scale'])

    def cbBenchmark(result):
        if False:
            return 10
        pprint(result)

    def ebBenchmark(err):
        if False:
            while True:
                i = 10
        print(err.getTraceback())
    d.addCallbacks(cbBenchmark, ebBenchmark)

    def stopReactor(ign):
        if False:
            for i in range(10):
                print('nop')
        reactor.stop()
    d.addBoth(stopReactor)
    reactor.run()
if __name__ == '__main__':
    main()