"""
Producer-Consumer Proxy.
"""
from zope.interface import implementer
from twisted.internet import interfaces

@implementer(interfaces.IProducer, interfaces.IConsumer)
class BasicProducerConsumerProxy:
    """
    I can act as a man in the middle between any Producer and Consumer.

    @ivar producer: the Producer I subscribe to.
    @type producer: L{IProducer<interfaces.IProducer>}
    @ivar consumer: the Consumer I publish to.
    @type consumer: L{IConsumer<interfaces.IConsumer>}
    @ivar paused: As a Producer, am I paused?
    @type paused: bool
    """
    consumer = None
    producer = None
    producerIsStreaming = None
    iAmStreaming = True
    outstandingPull = False
    paused = False
    stopped = False

    def __init__(self, consumer):
        if False:
            for i in range(10):
                print('nop')
        self._buffer = []
        if consumer is not None:
            self.consumer = consumer
            consumer.registerProducer(self, self.iAmStreaming)

    def pauseProducing(self):
        if False:
            return 10
        self.paused = True
        if self.producer:
            self.producer.pauseProducing()

    def resumeProducing(self):
        if False:
            i = 10
            return i + 15
        self.paused = False
        if self._buffer:
            self.consumer.write(''.join(self._buffer))
            self._buffer[:] = []
        elif not self.iAmStreaming:
            self.outstandingPull = True
        if self.producer is not None:
            self.producer.resumeProducing()

    def stopProducing(self):
        if False:
            return 10
        if self.producer is not None:
            self.producer.stopProducing()
        if self.consumer is not None:
            del self.consumer

    def write(self, data):
        if False:
            while True:
                i = 10
        if self.paused or (not self.iAmStreaming and (not self.outstandingPull)):
            self._buffer.append(data)
        elif self.consumer is not None:
            self.consumer.write(data)
            self.outstandingPull = False

    def finish(self):
        if False:
            i = 10
            return i + 15
        if self.consumer is not None:
            self.consumer.finish()
        self.unregisterProducer()

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        self.producer = producer
        self.producerIsStreaming = streaming

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        if self.producer is not None:
            del self.producer
            del self.producerIsStreaming
        if self.consumer:
            self.consumer.unregisterProducer()

    def __repr__(self) -> str:
        if False:
            return 10
        return f'<{self.__class__}@{id(self):x} around {self.consumer}>'

class ProducerConsumerProxy(BasicProducerConsumerProxy):
    """ProducerConsumerProxy with a finite buffer.

    When my buffer fills up, I have my parent Producer pause until my buffer
    has room in it again.
    """
    bufferSize = 2 ** 2 ** 2 ** 2
    producerPaused = False
    unregistered = False

    def pauseProducing(self):
        if False:
            return 10
        self.paused = True

    def resumeProducing(self):
        if False:
            print('Hello World!')
        self.paused = False
        if self._buffer:
            data = ''.join(self._buffer)
            bytesSent = self._writeSomeData(data)
            if bytesSent < len(data):
                unsent = data[bytesSent:]
                assert not self.iAmStreaming, 'Streaming producer did not write all its data.'
                self._buffer[:] = [unsent]
            else:
                self._buffer[:] = []
        else:
            bytesSent = 0
        if self.unregistered and bytesSent and (not self._buffer) and (self.consumer is not None):
            self.consumer.unregisterProducer()
        if not self.iAmStreaming:
            self.outstandingPull = not bytesSent
        if self.producer is not None:
            bytesBuffered = sum((len(s) for s in self._buffer))
            if self.producerPaused and bytesBuffered < self.bufferSize:
                self.producerPaused = False
                self.producer.resumeProducing()
            elif self.outstandingPull:
                self.producer.resumeProducing()

    def write(self, data):
        if False:
            while True:
                i = 10
        if self.paused or (not self.iAmStreaming and (not self.outstandingPull)):
            self._buffer.append(data)
        elif self.consumer is not None:
            assert not self._buffer, 'Writing fresh data to consumer before my buffer is empty!'
            bytesSent = self._writeSomeData(data)
            self.outstandingPull = False
            if not bytesSent == len(data):
                assert not self.iAmStreaming, 'Streaming producer did not write all its data.'
                self._buffer.append(data[bytesSent:])
        if self.producer is not None and self.producerIsStreaming:
            bytesBuffered = sum((len(s) for s in self._buffer))
            if bytesBuffered >= self.bufferSize:
                self.producer.pauseProducing()
                self.producerPaused = True

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        self.unregistered = False
        BasicProducerConsumerProxy.registerProducer(self, producer, streaming)
        if not streaming:
            producer.resumeProducing()

    def unregisterProducer(self):
        if False:
            i = 10
            return i + 15
        if self.producer is not None:
            del self.producer
            del self.producerIsStreaming
        self.unregistered = True
        if self.consumer and (not self._buffer):
            self.consumer.unregisterProducer()

    def _writeSomeData(self, data):
        if False:
            return 10
        'Write as much of this data as possible.\n\n        @returns: The number of bytes written.\n        '
        if self.consumer is None:
            return 0
        self.consumer.write(data)
        return len(data)