from __future__ import absolute_import
import six
from st2common.util import concurrency
__all__ = ['ConnectionRetryWrapper', 'ClusterRetryContext']

class ClusterRetryContext(object):
    """
    Stores retry context for cluster retries. It makes certain assumptions
    on how cluster_size and retry should be determined.
    """

    def __init__(self, cluster_size):
        if False:
            while True:
                i = 10
        self.cluster_size = cluster_size
        self.cluster_retry = 2
        self.wait_between_cluster = 10
        self._nodes_attempted = 1

    def test_should_stop(self, e=None):
        if False:
            while True:
                i = 10
        if "second 'channel.open' seen" in six.text_type(e):
            return (False, -1)
        should_stop = True
        if self._nodes_attempted > self.cluster_size * self.cluster_retry:
            return (should_stop, -1)
        wait = 0
        should_stop = False
        if self._nodes_attempted % self.cluster_size == 0:
            wait = self.wait_between_cluster
        self._nodes_attempted += 1
        return (should_stop, wait)

class ConnectionRetryWrapper(object):
    """
    Manages retry of connection and also switching to different nodes in a cluster.

    :param cluster_size: Size of the cluster.
    :param logger: logger to use to log moderately useful information.

    .. code-block:: python
        # Without ensuring recoverable errors are retried
        connection_urls = [
            'amqp://guest:guest@node1:5672',
            'amqp://guest:guest@node2:5672',
            'amqp://guest:guest@node3:5672'
        ]
        with Connection(connection_urls) as connection:
            retry_wrapper = ConnectionRetryWrapper(cluster_size=len(connection_urls),
                                                   logger=my_logger)
            # wrapped_callback must have signature ``def func(connection, channel)``
            def wrapped_callback(connection, channel):
                pass

            retry_wrapper.run(connection=connection, wrapped_callback=wrapped_callback)

        # With ensuring recoverable errors are retried
        connection_urls = [
            'amqp://guest:guest@node1:5672',
            'amqp://guest:guest@node2:5672',
            'amqp://guest:guest@node3:5672'
        ]
        with Connection(connection_urls) as connection:
            retry_wrapper = ConnectionRetryWrapper(cluster_size=len(connection_urls),
                                                   logger=my_logger)
            # wrapped_callback must have signature ``def func(connection, channel)``
            def wrapped_callback(connection, channel):
                kwargs = {...}
                # call ensured to correctly deal with recoverable errors.
                retry_wrapper.ensured(connection=connection_retry_wrapper,
                                      obj=my_obj,
                                      to_ensure_func=my_obj.ensuree,
                                      **kwargs)

            retry_wrapper.run(connection=connection, wrapped_callback=wrapped_callback)

    """

    def __init__(self, cluster_size, logger, ensure_max_retries=3):
        if False:
            return 10
        self._retry_context = ClusterRetryContext(cluster_size=cluster_size)
        self._logger = logger
        self._ensure_max_retries = ensure_max_retries

    def errback(self, exc, interval):
        if False:
            return 10
        self._logger.error('Rabbitmq connection error: %s', exc.message)

    def run(self, connection, wrapped_callback):
        if False:
            i = 10
            return i + 15
        '\n        Run the wrapped_callback in a protective covering of retries and error handling.\n\n        :param connection: Connection to messaging service\n        :type connection: kombu.connection.Connection\n\n        :param wrapped_callback: Callback that will be wrapped by all the fine handling in this\n                                 method. Expected signature of callback -\n                                 ``def func(connection, channel)``\n        '
        should_stop = False
        channel = None
        while not should_stop:
            try:
                channel = connection.channel()
                wrapped_callback(connection=connection, channel=channel)
                should_stop = True
            except connection.connection_errors + connection.channel_errors as e:
                (should_stop, wait) = self._retry_context.test_should_stop(e)
                channel = None
                if should_stop:
                    raise
                self._logger.debug('Received RabbitMQ server error, sleeping for %s seconds before retrying: %s' % (wait, six.text_type(e)))
                concurrency.sleep(wait)
                connection.close()

                def log_error_on_conn_failure(exc, interval):
                    if False:
                        return 10
                    self._logger.debug('Failed to re-establish connection to RabbitMQ server, retrying in %s seconds: %s' % (interval, six.text_type(exc)))
                try:
                    connection.ensure_connection(max_retries=self._ensure_max_retries, errback=log_error_on_conn_failure)
                except Exception:
                    self._logger.exception('Connections to RabbitMQ cannot be re-established: %s', six.text_type(e))
                    raise
            except Exception as e:
                self._logger.exception('Connections to RabbitMQ cannot be re-established: %s', six.text_type(e))
                raise
            finally:
                if should_stop and channel:
                    try:
                        channel.close()
                    except Exception:
                        self._logger.warning('Error closing channel.', exc_info=True)

    def ensured(self, connection, obj, to_ensure_func, **kwargs):
        if False:
            return 10
        '\n        Ensure that recoverable errors are retried a set number of times before giving up.\n\n        :param connection: Connection to messaging service\n        :type connection: kombu.connection.Connection\n\n        :param obj: Object whose method is to be ensured. Typically, channel, producer etc. from\n                    the kombu library.\n        :type obj: Must support mixin kombu.abstract.MaybeChannelBound\n        '
        ensuring_func = connection.ensure(obj, to_ensure_func, errback=self.errback, max_retries=3)
        ensuring_func(**kwargs)