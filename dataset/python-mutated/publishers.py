from __future__ import absolute_import
import copy
from kombu.messaging import Producer
from oslo_config import cfg
from st2common import log as logging
from st2common.metrics.base import Timer
from st2common.transport import utils as transport_utils
from st2common.transport.connection_retry_wrapper import ConnectionRetryWrapper
__all__ = ['PoolPublisher', 'SharedPoolPublishers', 'CUDPublisher', 'StatePublisherMixin']
ANY_RK = '*'
CREATE_RK = 'create'
UPDATE_RK = 'update'
DELETE_RK = 'delete'
LOG = logging.getLogger(__name__)

class PoolPublisher(object):

    def __init__(self, urls=None):
        if False:
            while True:
                i = 10
        '\n        :param urls: Connection URLs to use. If not provided it uses a default value from th\n                     config.\n        :type urls: ``list``\n        '
        urls = urls or transport_utils.get_messaging_urls()
        connection = transport_utils.get_connection(urls=urls, connection_kwargs={'failover_strategy': 'round-robin'})
        self.pool = connection.Pool(limit=10)
        self.cluster_size = len(urls)

    def errback(self, exc, interval):
        if False:
            for i in range(10):
                print('nop')
        LOG.error('Rabbitmq connection error: %s', exc.message, exc_info=False)

    def publish(self, payload, exchange, routing_key='', compression=None):
        if False:
            return 10
        compression = compression or cfg.CONF.messaging.compression
        with Timer(key='amqp.pool_publisher.publish_with_retries.' + exchange.name):
            with self.pool.acquire(block=True) as connection:
                retry_wrapper = ConnectionRetryWrapper(cluster_size=self.cluster_size, logger=LOG)

                def do_publish(connection, channel):
                    if False:
                        i = 10
                        return i + 15
                    producer = Producer(channel)
                    kwargs = {'body': payload, 'exchange': exchange, 'routing_key': routing_key, 'serializer': 'pickle', 'compression': compression, 'content_encoding': 'utf-8'}
                    retry_wrapper.ensured(connection=connection, obj=producer, to_ensure_func=producer.publish, **kwargs)
                retry_wrapper.run(connection=connection, wrapped_callback=do_publish)

class SharedPoolPublishers(object):
    """
    This maintains some shared PoolPublishers. Within a single process the configured AMQP
    server is usually the same. This sharing allows from the same PoolPublisher to be reused
    for publishing purposes. Sharing publishers leads to shared connections.
    """
    shared_publishers = {}

    def get_publisher(self, urls):
        if False:
            for i in range(10):
                print('nop')
        urls_copy = copy.copy(urls)
        urls_copy.sort()
        publisher_key = ''.join(urls_copy)
        publisher = self.shared_publishers.get(publisher_key, None)
        if not publisher:
            publisher = PoolPublisher(urls=urls)
            self.shared_publishers[publisher_key] = publisher
        return publisher

class CUDPublisher(object):

    def __init__(self, exchange):
        if False:
            print('Hello World!')
        urls = transport_utils.get_messaging_urls()
        self._publisher = SharedPoolPublishers().get_publisher(urls=urls)
        self._exchange = exchange

    def publish_create(self, payload):
        if False:
            return 10
        with Timer(key='amqp.publish.create'):
            self._publisher.publish(payload, self._exchange, CREATE_RK)

    def publish_update(self, payload):
        if False:
            for i in range(10):
                print('nop')
        with Timer(key='amqp.publish.update'):
            self._publisher.publish(payload, self._exchange, UPDATE_RK)

    def publish_delete(self, payload):
        if False:
            return 10
        with Timer(key='amqp.publish.delete'):
            self._publisher.publish(payload, self._exchange, DELETE_RK)

class StatePublisherMixin(object):

    def __init__(self, exchange):
        if False:
            print('Hello World!')
        urls = transport_utils.get_messaging_urls()
        self._state_publisher = SharedPoolPublishers().get_publisher(urls=urls)
        self._state_exchange = exchange

    def publish_state(self, payload, state):
        if False:
            print('Hello World!')
        if not state:
            raise Exception('Unable to publish unassigned state.')
        with Timer(key='amqp.publish.state'):
            self._state_publisher.publish(payload, self._state_exchange, state)