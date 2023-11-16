"""
Helpers for working with producers.
"""
from typing import List
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.task import cooperate
from twisted.python import log
from twisted.python.reflect import safe_str
__all__: List[str] = []

@implementer(IPushProducer)
class _PullToPush:
    """
    An adapter that converts a non-streaming to a streaming producer.

    Because of limitations of the producer API, this adapter requires the
    cooperation of the consumer. When the consumer's C{registerProducer} is
    called with a non-streaming producer, it must wrap it with L{_PullToPush}
    and then call C{startStreaming} on the resulting object. When the
    consumer's C{unregisterProducer} is called, it must call
    C{stopStreaming} on the L{_PullToPush} instance.

    If the underlying producer throws an exception from C{resumeProducing},
    the producer will be unregistered from the consumer.

    @ivar _producer: the underling non-streaming producer.

    @ivar _consumer: the consumer with which the underlying producer was
                     registered.

    @ivar _finished: C{bool} indicating whether the producer has finished.

    @ivar _coopTask: the result of calling L{cooperate}, the task driving the
                     streaming producer.
    """
    _finished = False

    def __init__(self, pullProducer, consumer):
        if False:
            print('Hello World!')
        self._producer = pullProducer
        self._consumer = consumer

    def _pull(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A generator that calls C{resumeProducing} on the underlying producer\n        forever.\n\n        If C{resumeProducing} throws an exception, the producer is\n        unregistered, which should result in streaming stopping.\n        '
        while True:
            try:
                self._producer.resumeProducing()
            except BaseException:
                log.err(None, '%s failed, producing will be stopped:' % (safe_str(self._producer),))
                try:
                    self._consumer.unregisterProducer()
                except BaseException:
                    log.err(None, '%s failed to unregister producer:' % (safe_str(self._consumer),))
                    self._finished = True
                    return
            yield None

    def startStreaming(self):
        if False:
            while True:
                i = 10
        '\n        This should be called by the consumer when the producer is registered.\n\n        Start streaming data to the consumer.\n        '
        self._coopTask = cooperate(self._pull())

    def stopStreaming(self):
        if False:
            i = 10
            return i + 15
        '\n        This should be called by the consumer when the producer is\n        unregistered.\n\n        Stop streaming data to the consumer.\n        '
        if self._finished:
            return
        self._finished = True
        self._coopTask.stop()

    def pauseProducing(self):
        if False:
            print('Hello World!')
        '\n        @see: C{IPushProducer.pauseProducing}\n        '
        self._coopTask.pause()

    def resumeProducing(self):
        if False:
            while True:
                i = 10
        '\n        @see: C{IPushProducer.resumeProducing}\n        '
        self._coopTask.resume()

    def stopProducing(self):
        if False:
            print('Hello World!')
        '\n        @see: C{IPushProducer.stopProducing}\n        '
        self.stopStreaming()
        self._producer.stopProducing()