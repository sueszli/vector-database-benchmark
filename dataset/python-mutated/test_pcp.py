__version__ = '$Revision: 1.5 $'[11:-2]
from twisted.protocols import pcp
from twisted.trial import unittest

class DummyTransport:
    """A dumb transport to wrap around."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._writes = []

    def write(self, data):
        if False:
            return 10
        self._writes.append(data)

    def getvalue(self):
        if False:
            return 10
        return ''.join(self._writes)

class DummyProducer:
    resumed = False
    stopped = False
    paused = False

    def __init__(self, consumer):
        if False:
            i = 10
            return i + 15
        self.consumer = consumer

    def resumeProducing(self):
        if False:
            print('Hello World!')
        self.resumed = True
        self.paused = False

    def pauseProducing(self):
        if False:
            print('Hello World!')
        self.paused = True

    def stopProducing(self):
        if False:
            return 10
        self.stopped = True

class DummyConsumer(DummyTransport):
    producer = None
    finished = False
    unregistered = True

    def registerProducer(self, producer, streaming):
        if False:
            i = 10
            return i + 15
        self.producer = (producer, streaming)

    def unregisterProducer(self):
        if False:
            i = 10
            return i + 15
        self.unregistered = True

    def finish(self):
        if False:
            return 10
        self.finished = True

class TransportInterfaceTests(unittest.TestCase):
    proxyClass = pcp.BasicProducerConsumerProxy

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.underlying = DummyConsumer()
        self.transport = self.proxyClass(self.underlying)

    def testWrite(self):
        if False:
            while True:
                i = 10
        self.transport.write('some bytes')

class ConsumerInterfaceTest:
    """Test ProducerConsumerProxy as a Consumer.

    Normally we have ProducingServer -> ConsumingTransport.

    If I am to go between (Server -> Shaper -> Transport), I have to
    play the role of Consumer convincingly for the ProducingServer.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.underlying = DummyConsumer()
        self.consumer = self.proxyClass(self.underlying)
        self.producer = DummyProducer(self.consumer)

    def testRegisterPush(self):
        if False:
            print('Hello World!')
        self.consumer.registerProducer(self.producer, True)
        self.assertFalse(self.producer.resumed)

    def testUnregister(self):
        if False:
            print('Hello World!')
        self.consumer.registerProducer(self.producer, False)
        self.consumer.unregisterProducer()
        self.producer.resumed = False
        self.consumer.resumeProducing()
        self.assertFalse(self.producer.resumed)

    def testFinish(self):
        if False:
            print('Hello World!')
        self.consumer.registerProducer(self.producer, False)
        self.consumer.finish()
        self.producer.resumed = False
        self.consumer.resumeProducing()
        self.assertFalse(self.producer.resumed)

class ProducerInterfaceTest:
    """Test ProducerConsumerProxy as a Producer.

    Normally we have ProducingServer -> ConsumingTransport.

    If I am to go between (Server -> Shaper -> Transport), I have to
    play the role of Producer convincingly for the ConsumingTransport.
    """

    def setUp(self):
        if False:
            return 10
        self.consumer = DummyConsumer()
        self.producer = self.proxyClass(self.consumer)

    def testRegistersProducer(self):
        if False:
            return 10
        self.assertEqual(self.consumer.producer[0], self.producer)

    def testPause(self):
        if False:
            print('Hello World!')
        self.producer.pauseProducing()
        self.producer.write('yakkity yak')
        self.assertFalse(self.consumer.getvalue(), 'Paused producer should not have sent data.')

    def testResume(self):
        if False:
            while True:
                i = 10
        self.producer.pauseProducing()
        self.producer.resumeProducing()
        self.producer.write('yakkity yak')
        self.assertEqual(self.consumer.getvalue(), 'yakkity yak')

    def testResumeNoEmptyWrite(self):
        if False:
            print('Hello World!')
        self.producer.pauseProducing()
        self.producer.resumeProducing()
        self.assertEqual(len(self.consumer._writes), 0, 'Resume triggered an empty write.')

    def testResumeBuffer(self):
        if False:
            i = 10
            return i + 15
        self.producer.pauseProducing()
        self.producer.write('buffer this')
        self.producer.resumeProducing()
        self.assertEqual(self.consumer.getvalue(), 'buffer this')

    def testStop(self):
        if False:
            i = 10
            return i + 15
        self.producer.stopProducing()
        self.producer.write('yakkity yak')
        self.assertFalse(self.consumer.getvalue(), 'Stopped producer should not have sent data.')

class PCP_ConsumerInterfaceTests(ConsumerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.BasicProducerConsumerProxy

class PCPII_ConsumerInterfaceTests(ConsumerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.ProducerConsumerProxy

class PCP_ProducerInterfaceTests(ProducerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.BasicProducerConsumerProxy

class PCPII_ProducerInterfaceTests(ProducerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.ProducerConsumerProxy

class ProducerProxyTests(unittest.TestCase):
    """Producer methods on me should be relayed to the Producer I proxy."""
    proxyClass = pcp.BasicProducerConsumerProxy

    def setUp(self):
        if False:
            return 10
        self.proxy = self.proxyClass(None)
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, True)

    def testStop(self):
        if False:
            return 10
        self.proxy.stopProducing()
        self.assertTrue(self.parentProducer.stopped)

class ConsumerProxyTests(unittest.TestCase):
    """Consumer methods on me should be relayed to the Consumer I proxy."""
    proxyClass = pcp.BasicProducerConsumerProxy

    def setUp(self):
        if False:
            print('Hello World!')
        self.underlying = DummyConsumer()
        self.consumer = self.proxyClass(self.underlying)

    def testWrite(self):
        if False:
            while True:
                i = 10
        self.consumer.write('some bytes')
        self.assertEqual(self.underlying.getvalue(), 'some bytes')

    def testFinish(self):
        if False:
            i = 10
            return i + 15
        self.consumer.finish()
        self.assertTrue(self.underlying.finished)

    def testUnregister(self):
        if False:
            return 10
        self.consumer.unregisterProducer()
        self.assertTrue(self.underlying.unregistered)

class PullProducerTest:

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.underlying = DummyConsumer()
        self.proxy = self.proxyClass(self.underlying)
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, True)

    def testHoldWrites(self):
        if False:
            return 10
        self.proxy.write('hello')
        self.assertFalse(self.underlying.getvalue(), 'Pulling Consumer got data before it pulled.')

    def testPull(self):
        if False:
            return 10
        self.proxy.write('hello')
        self.proxy.resumeProducing()
        self.assertEqual(self.underlying.getvalue(), 'hello')

    def testMergeWrites(self):
        if False:
            print('Hello World!')
        self.proxy.write('hello ')
        self.proxy.write('sunshine')
        self.proxy.resumeProducing()
        nwrites = len(self.underlying._writes)
        self.assertEqual(nwrites, 1, 'Pull resulted in %d writes instead of 1.' % (nwrites,))
        self.assertEqual(self.underlying.getvalue(), 'hello sunshine')

    def testLateWrite(self):
        if False:
            while True:
                i = 10
        self.proxy.resumeProducing()
        self.proxy.write('data')
        self.assertEqual(self.underlying.getvalue(), 'data')

class PCP_PullProducerTests(PullProducerTest, unittest.TestCase):

    class proxyClass(pcp.BasicProducerConsumerProxy):
        iAmStreaming = False

class PCPII_PullProducerTests(PullProducerTest, unittest.TestCase):

    class proxyClass(pcp.ProducerConsumerProxy):
        iAmStreaming = False

class BufferedConsumerTests(unittest.TestCase):
    """As a consumer, ask the producer to pause after too much data."""
    proxyClass = pcp.ProducerConsumerProxy

    def setUp(self):
        if False:
            print('Hello World!')
        self.underlying = DummyConsumer()
        self.proxy = self.proxyClass(self.underlying)
        self.proxy.bufferSize = 100
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, True)

    def testRegisterPull(self):
        if False:
            i = 10
            return i + 15
        self.proxy.registerProducer(self.parentProducer, False)
        self.assertTrue(self.parentProducer.resumed)

    def testPauseIntercept(self):
        if False:
            return 10
        self.proxy.pauseProducing()
        self.assertFalse(self.parentProducer.paused)

    def testResumeIntercept(self):
        if False:
            print('Hello World!')
        self.proxy.pauseProducing()
        self.proxy.resumeProducing()
        self.assertFalse(self.parentProducer.resumed)

    def testTriggerPause(self):
        if False:
            i = 10
            return i + 15
        'Make sure I say "when." '
        self.proxy.pauseProducing()
        self.assertFalse(self.parentProducer.paused, "don't pause yet")
        self.proxy.write('x' * 51)
        self.assertFalse(self.parentProducer.paused, "don't pause yet")
        self.proxy.write('x' * 51)
        self.assertTrue(self.parentProducer.paused)

    def testTriggerResume(self):
        if False:
            print('Hello World!')
        'Make sure I resumeProducing when my buffer empties.'
        self.proxy.pauseProducing()
        self.proxy.write('x' * 102)
        self.assertTrue(self.parentProducer.paused, 'should be paused')
        self.proxy.resumeProducing()
        self.assertFalse(self.parentProducer.paused, 'Producer should have resumed.')
        self.assertFalse(self.proxy.producerPaused)

class BufferedPullTests(unittest.TestCase):

    class proxyClass(pcp.ProducerConsumerProxy):
        iAmStreaming = False

        def _writeSomeData(self, data):
            if False:
                return 10
            pcp.ProducerConsumerProxy._writeSomeData(self, data[:100])
            return min(len(data), 100)

    def setUp(self):
        if False:
            print('Hello World!')
        self.underlying = DummyConsumer()
        self.proxy = self.proxyClass(self.underlying)
        self.proxy.bufferSize = 100
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, False)

    def testResumePull(self):
        if False:
            print('Hello World!')
        self.parentProducer.resumed = False
        self.proxy.resumeProducing()
        self.assertTrue(self.parentProducer.resumed)

    def testLateWriteBuffering(self):
        if False:
            while True:
                i = 10
        self.proxy.resumeProducing()
        self.proxy.write('datum' * 21)
        self.assertEqual(self.underlying.getvalue(), 'datum' * 20)
        self.assertEqual(self.proxy._buffer, ['datum'])