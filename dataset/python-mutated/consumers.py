from __future__ import absolute_import
import abc
import six
from kombu.mixins import ConsumerMixin
from oslo_config import cfg
from st2common import log as logging
from st2common.util.greenpooldispatch import BufferedDispatcher
from st2common.util import concurrency
__all__ = ['QueueConsumer', 'StagedQueueConsumer', 'ActionsQueueConsumer', 'MessageHandler', 'StagedMessageHandler']
LOG = logging.getLogger(__name__)

class QueueConsumer(ConsumerMixin):

    def __init__(self, connection, queues, handler):
        if False:
            while True:
                i = 10
        self.connection = connection
        self._dispatcher = BufferedDispatcher()
        self._queues = queues
        self._handler = handler

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        self.should_stop = True
        self._dispatcher.shutdown()

    def get_consumers(self, Consumer, channel):
        if False:
            print('Hello World!')
        consumer = Consumer(queues=self._queues, accept=['pickle'], callbacks=[self.process])
        consumer.qos(prefetch_count=1)
        return [consumer]

    def process(self, body, message):
        if False:
            i = 10
            return i + 15
        try:
            if not isinstance(body, self._handler.message_type):
                raise TypeError('Received an unexpected type "%s" for payload.' % type(body))
            self._dispatcher.dispatch(self._process_message, body)
        except:
            LOG.exception('%s failed to process message: %s', self.__class__.__name__, body)
        finally:
            message.ack()

    def _process_message(self, body):
        if False:
            print('Hello World!')
        try:
            self._handler.process(body)
        except:
            LOG.exception('%s failed to process message: %s', self.__class__.__name__, body)

class StagedQueueConsumer(QueueConsumer):
    """
    Used by ``StagedMessageHandler`` to effectively manage it 2 step message handling.
    """

    def process(self, body, message):
        if False:
            i = 10
            return i + 15
        try:
            if not isinstance(body, self._handler.message_type):
                raise TypeError('Received an unexpected type "%s" for payload.' % type(body))
            response = self._handler.pre_ack_process(body)
            self._dispatcher.dispatch(self._process_message, response)
        except:
            LOG.exception('%s failed to process message: %s', self.__class__.__name__, body)
        finally:
            message.ack()

class ActionsQueueConsumer(QueueConsumer):
    """
    Special Queue Consumer for action runner which uses multiple BufferedDispatcher pools:

    1. For regular (non-workflow) actions
    2. One for workflow actions

    This way we can ensure workflow actions never block non-workflow actions.
    """

    def __init__(self, connection, queues, handler):
        if False:
            return 10
        self.connection = connection
        self._queues = queues
        self._handler = handler
        workflows_pool_size = cfg.CONF.actionrunner.workflows_pool_size
        actions_pool_size = cfg.CONF.actionrunner.actions_pool_size
        self._workflows_dispatcher = BufferedDispatcher(dispatch_pool_size=workflows_pool_size, name='workflows-dispatcher')
        self._actions_dispatcher = BufferedDispatcher(dispatch_pool_size=actions_pool_size, name='actions-dispatcher')

    def process(self, body, message):
        if False:
            print('Hello World!')
        try:
            if not isinstance(body, self._handler.message_type):
                raise TypeError('Received an unexpected type "%s" for payload.' % type(body))
            action_is_workflow = getattr(body, 'action_is_workflow', False)
            if action_is_workflow:
                dispatcher = self._workflows_dispatcher
            else:
                dispatcher = self._actions_dispatcher
            LOG.debug('Using BufferedDispatcher pool: "%s"', str(dispatcher))
            dispatcher.dispatch(self._process_message, body)
        except:
            LOG.exception('%s failed to process message: %s', self.__class__.__name__, body)
        finally:
            message.ack()

    def shutdown(self):
        if False:
            while True:
                i = 10
        self._workflows_dispatcher.shutdown()
        self._actions_dispatcher.shutdown()
        self.should_stop = True

class VariableMessageQueueConsumer(QueueConsumer):
    """
    Used by ``VariableMessageHandler`` to processes multiple message types.
    """

    def process(self, body, message):
        if False:
            i = 10
            return i + 15
        try:
            if not self._handler.message_types.get(type(body)):
                raise TypeError('Received an unexpected type "%s" for payload.' % type(body))
            self._dispatcher.dispatch(self._process_message, body)
        except:
            LOG.exception('%s failed to process message: %s', self.__class__.__name__, body)
        finally:
            message.ack()

@six.add_metaclass(abc.ABCMeta)
class MessageHandler(object):
    message_type = None

    def __init__(self, connection, queues):
        if False:
            while True:
                i = 10
        self._queue_consumer = self.get_queue_consumer(connection=connection, queues=queues)
        self._consumer_thread = None

    def start(self, wait=False):
        if False:
            print('Hello World!')
        LOG.info('Starting %s...', self.__class__.__name__)
        self._consumer_thread = concurrency.spawn(self._queue_consumer.run)
        if wait:
            self.wait()

    def wait(self):
        if False:
            return 10
        self._consumer_thread.wait()

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        LOG.info('Shutting down %s...', self.__class__.__name__)
        self._queue_consumer.shutdown()

    def kill(self):
        if False:
            i = 10
            return i + 15
        self._consumer_thread.kill(SystemExit())

    @abc.abstractmethod
    def process(self, message):
        if False:
            i = 10
            return i + 15
        pass

    def get_queue_consumer(self, connection, queues):
        if False:
            for i in range(10):
                print('nop')
        return QueueConsumer(connection=connection, queues=queues, handler=self)

@six.add_metaclass(abc.ABCMeta)
class StagedMessageHandler(MessageHandler):
    """
    MessageHandler to deal with messages in 2 steps.
        1. pre_ack_process : This is called on the handler before ack-ing the message.
        2. process: Called after ack-in the messages
    This 2 step approach provides a way for the handler to do some hadling like saving to DB etc
    before acknowleding and then performing future processing async. This way even if the handler
    or owning process is taken down system will still maintain track of the message.
    """

    @abc.abstractmethod
    def pre_ack_process(self, message):
        if False:
            while True:
                i = 10
        '\n        Called before acknowleding a message. Good place to track the message via a DB entry or some\n        other applicable mechnism.\n\n        The reponse of this method is passed into the ``process`` method. This was whatever is the\n        processed version of the message can be moved forward. It is always possible to simply\n        return ``message`` and have ``process`` handle the original message.\n        '
        pass

    def get_queue_consumer(self, connection, queues):
        if False:
            while True:
                i = 10
        return StagedQueueConsumer(connection=connection, queues=queues, handler=self)

@six.add_metaclass(abc.ABCMeta)
class VariableMessageHandler(MessageHandler):
    """
    VariableMessageHandler processes multiple message types.
    """

    def get_queue_consumer(self, connection, queues):
        if False:
            return 10
        return VariableMessageQueueConsumer(connection=connection, queues=queues, handler=self)