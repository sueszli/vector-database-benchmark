from __future__ import absolute_import
import six
from kombu.mixins import ConsumerMixin
from st2common import log as logging
from st2common.transport import reactor, publishers
from st2common.transport import utils as transport_utils
from st2common.util import concurrency
import st2common.util.queues as queue_utils
LOG = logging.getLogger(__name__)

class SensorWatcher(ConsumerMixin):

    def __init__(self, create_handler, update_handler, delete_handler, queue_suffix=None):
        if False:
            print('Hello World!')
        '\n        :param create_handler: Function which is called on SensorDB create event.\n        :type create_handler: ``callable``\n\n        :param update_handler: Function which is called on SensorDB update event.\n        :type update_handler: ``callable``\n\n        :param delete_handler: Function which is called on SensorDB delete event.\n        :type delete_handler: ``callable``\n        '
        self._create_handler = create_handler
        self._update_handler = update_handler
        self._delete_handler = delete_handler
        self._sensor_watcher_q = self._get_queue(queue_suffix)
        self.connection = None
        self._updates_thread = None
        self._handlers = {publishers.CREATE_RK: create_handler, publishers.UPDATE_RK: update_handler, publishers.DELETE_RK: delete_handler}

    def get_consumers(self, Consumer, channel):
        if False:
            for i in range(10):
                print('nop')
        consumers = [Consumer(queues=[self._sensor_watcher_q], accept=['pickle'], callbacks=[self.process_task])]
        return consumers

    def process_task(self, body, message):
        if False:
            i = 10
            return i + 15
        LOG.debug('process_task')
        LOG.debug('     body: %s', body)
        LOG.debug('     message.properties: %s', message.properties)
        LOG.debug('     message.delivery_info: %s', message.delivery_info)
        routing_key = message.delivery_info.get('routing_key', '')
        handler = self._handlers.get(routing_key, None)
        try:
            if not handler:
                LOG.info('Skipping message %s as no handler was found.', message)
                return
            try:
                handler(body)
            except Exception as e:
                LOG.exception('Handling failed. Message body: %s. Exception: %s', body, six.text_type(e))
        finally:
            message.ack()

    def start(self):
        if False:
            while True:
                i = 10
        try:
            self.connection = transport_utils.get_connection()
            self._updates_thread = concurrency.spawn(self.run)
        except:
            LOG.exception('Failed to start sensor_watcher.')
            self.connection.release()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        LOG.debug('Shutting down sensor watcher.')
        try:
            if self._updates_thread:
                self._updates_thread = concurrency.kill(self._updates_thread)
            if self.connection:
                channel = self.connection.channel()
                bound_sensor_watch_q = self._sensor_watcher_q(channel)
                try:
                    bound_sensor_watch_q.delete()
                except:
                    LOG.error('Unable to delete sensor watcher queue: %s', self._sensor_watcher_q)
        finally:
            if self.connection:
                self.connection.release()

    @staticmethod
    def _get_queue(queue_suffix):
        if False:
            print('Hello World!')
        queue_name = queue_utils.get_queue_name(queue_name_base='st2.sensor.watch', queue_name_suffix=queue_suffix, add_random_uuid_to_suffix=True)
        return reactor.get_sensor_cud_queue(queue_name, routing_key='#')