from __future__ import absolute_import
import six
from kombu.mixins import ConsumerMixin
from st2common import log as logging
from st2common.persistence.trigger import Trigger
from st2common.transport import reactor, publishers
from st2common.transport import utils as transport_utils
from st2common.util import concurrency
import st2common.util.queues as queue_utils
LOG = logging.getLogger(__name__)

class TriggerWatcher(ConsumerMixin):
    sleep_interval = 0

    def __init__(self, create_handler, update_handler, delete_handler, trigger_types=None, queue_suffix=None, exclusive=False):
        if False:
            print('Hello World!')
        '\n        :param create_handler: Function which is called on TriggerDB create event.\n        :type create_handler: ``callable``\n\n        :param update_handler: Function which is called on TriggerDB update event.\n        :type update_handler: ``callable``\n\n        :param delete_handler: Function which is called on TriggerDB delete event.\n        :type delete_handler: ``callable``\n\n        :param trigger_types: If provided, handler function will only be called\n                              if the trigger in the message payload is included\n                              in this list.\n        :type trigger_types: ``list``\n\n        :param exclusive: If the Q is exclusive to a specific connection which is then\n                          single connection created by TriggerWatcher. When the connection\n                          breaks the Q is removed by the message broker.\n        :type exclusive: ``bool``\n        '
        self._create_handler = create_handler
        self._update_handler = update_handler
        self._delete_handler = delete_handler
        self._trigger_types = trigger_types
        self._trigger_watch_q = self._get_queue(queue_suffix, exclusive=exclusive)
        self.connection = None
        self._load_thread = None
        self._updates_thread = None
        self._handlers = {publishers.CREATE_RK: create_handler, publishers.UPDATE_RK: update_handler, publishers.DELETE_RK: delete_handler}

    def get_consumers(self, Consumer, channel):
        if False:
            for i in range(10):
                print('nop')
        return [Consumer(queues=[self._trigger_watch_q], accept=['pickle'], callbacks=[self.process_task])]

    def process_task(self, body, message):
        if False:
            print('Hello World!')
        LOG.debug('process_task')
        LOG.debug('     body: %s', body)
        LOG.debug('     message.properties: %s', message.properties)
        LOG.debug('     message.delivery_info: %s', message.delivery_info)
        routing_key = message.delivery_info.get('routing_key', '')
        handler = self._handlers.get(routing_key, None)
        try:
            if not handler:
                LOG.debug('Skipping message %s as no handler was found.', message)
                return
            trigger_type = getattr(body, 'type', None)
            if self._trigger_types and trigger_type not in self._trigger_types:
                LOG.debug("Skipping message %s since trigger_type doesn't match (type=%s)", message, trigger_type)
                return
            try:
                handler(body)
            except Exception as e:
                LOG.exception('Handling failed. Message body: %s. Exception: %s', body, six.text_type(e))
        finally:
            message.ack()
        concurrency.sleep(self.sleep_interval)

    def start(self):
        if False:
            print('Hello World!')
        try:
            self.connection = transport_utils.get_connection()
            self._updates_thread = concurrency.spawn(self.run)
            self._load_thread = concurrency.spawn(self._load_triggers_from_db)
        except:
            LOG.exception('Failed to start watcher.')
            self.connection.release()

    def stop(self):
        if False:
            return 10
        try:
            self._updates_thread = concurrency.kill(self._updates_thread)
            self._load_thread = concurrency.kill(self._load_thread)
        finally:
            self.connection.release()

    def on_consume_end(self, connection, channel):
        if False:
            for i in range(10):
                print('nop')
        super(TriggerWatcher, self).on_consume_end(connection=connection, channel=channel)
        concurrency.sleep(seconds=self.sleep_interval)

    def on_iteration(self):
        if False:
            for i in range(10):
                print('nop')
        super(TriggerWatcher, self).on_iteration()
        concurrency.sleep(seconds=self.sleep_interval)

    def _load_triggers_from_db(self):
        if False:
            print('Hello World!')
        for trigger_type in self._trigger_types:
            for trigger in Trigger.query(type=trigger_type):
                LOG.debug('Found existing trigger: %s in db.' % trigger)
                self._handlers[publishers.CREATE_RK](trigger)

    @staticmethod
    def _get_queue(queue_suffix, exclusive):
        if False:
            print('Hello World!')
        queue_name = queue_utils.get_queue_name(queue_name_base='st2.trigger.watch', queue_name_suffix=queue_suffix, add_random_uuid_to_suffix=True)
        return reactor.get_trigger_cud_queue(queue_name, routing_key='#', exclusive=exclusive)