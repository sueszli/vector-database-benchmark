import functools
import logging
import time
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from pika.exchange_type import ExchangeType
LOG_FORMAT = '%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s'
LOGGER = logging.getLogger(__name__)

class ExampleConsumer(object):
    """This is an example consumer that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, this class will stop and indicate
    that reconnection is necessary. You should look at the output, as
    there are limited reasons why the connection may be closed, which
    usually are tied to permission related issues or socket timeouts.

    If the channel is closed, it will indicate a problem with one of the
    commands that were issued and that should surface in the output as well.

    """
    EXCHANGE = 'message'
    EXCHANGE_TYPE = ExchangeType.topic
    QUEUE = 'text'
    ROUTING_KEY = 'example.text'

    def __init__(self, amqp_url):
        if False:
            print('Hello World!')
        'Create a new instance of the consumer class, passing in the AMQP\n        URL used to connect to RabbitMQ.\n\n        :param str amqp_url: The AMQP url to connect with\n\n        '
        self.should_reconnect = False
        self.was_consuming = False
        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._url = amqp_url
        self._consuming = False
        self._prefetch_count = 1

    def connect(self):
        if False:
            for i in range(10):
                print('nop')
        'This method connects to RabbitMQ, returning the connection handle.\n        When the connection is established, the on_connection_open method\n        will be invoked by pika.\n\n        :rtype: pika.adapters.asyncio_connection.AsyncioConnection\n\n        '
        LOGGER.info('Connecting to %s', self._url)
        return AsyncioConnection(parameters=pika.URLParameters(self._url), on_open_callback=self.on_connection_open, on_open_error_callback=self.on_connection_open_error, on_close_callback=self.on_connection_closed)

    def close_connection(self):
        if False:
            return 10
        self._consuming = False
        if self._connection.is_closing or self._connection.is_closed:
            LOGGER.info('Connection is closing or already closed')
        else:
            LOGGER.info('Closing connection')
            self._connection.close()

    def on_connection_open(self, _unused_connection):
        if False:
            for i in range(10):
                print('nop')
        "This method is called by pika once the connection to RabbitMQ has\n        been established. It passes the handle to the connection object in\n        case we need it, but in this case, we'll just mark it unused.\n\n        :param pika.adapters.asyncio_connection.AsyncioConnection _unused_connection:\n           The connection\n\n        "
        LOGGER.info('Connection opened')
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        if False:
            for i in range(10):
                print('nop')
        "This method is called by pika if the connection to RabbitMQ\n        can't be established.\n\n        :param pika.adapters.asyncio_connection.AsyncioConnection _unused_connection:\n           The connection\n        :param Exception err: The error\n\n        "
        LOGGER.error('Connection open failed: %s', err)
        self.reconnect()

    def on_connection_closed(self, _unused_connection, reason):
        if False:
            for i in range(10):
                print('nop')
        'This method is invoked by pika when the connection to RabbitMQ is\n        closed unexpectedly. Since it is unexpected, we will reconnect to\n        RabbitMQ if it disconnects.\n\n        :param pika.connection.Connection connection: The closed connection obj\n        :param Exception reason: exception representing reason for loss of\n            connection.\n\n        '
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            LOGGER.warning('Connection closed, reconnect necessary: %s', reason)
            self.reconnect()

    def reconnect(self):
        if False:
            i = 10
            return i + 15
        "Will be invoked if the connection can't be opened or is\n        closed. Indicates that a reconnect is necessary then stops the\n        ioloop.\n\n        "
        self.should_reconnect = True
        self.stop()

    def open_channel(self):
        if False:
            while True:
                i = 10
        'Open a new channel with RabbitMQ by issuing the Channel.Open RPC\n        command. When RabbitMQ responds that the channel is open, the\n        on_channel_open callback will be invoked by pika.\n\n        '
        LOGGER.info('Creating a new channel')
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        if False:
            print('Hello World!')
        "This method is invoked by pika when the channel has been opened.\n        The channel object is passed in so we can make use of it.\n\n        Since the channel is now open, we'll declare the exchange to use.\n\n        :param pika.channel.Channel channel: The channel object\n\n        "
        LOGGER.info('Channel opened')
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.EXCHANGE)

    def add_on_channel_close_callback(self):
        if False:
            return 10
        'This method tells pika to call the on_channel_closed method if\n        RabbitMQ unexpectedly closes the channel.\n\n        '
        LOGGER.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        if False:
            print('Hello World!')
        "Invoked by pika when RabbitMQ unexpectedly closes the channel.\n        Channels are usually closed if you attempt to do something that\n        violates the protocol, such as re-declare an exchange or queue with\n        different parameters. In this case, we'll close the connection\n        to shutdown the object.\n\n        :param pika.channel.Channel: The closed channel\n        :param Exception reason: why the channel was closed\n\n        "
        LOGGER.warning('Channel %i was closed: %s', channel, reason)
        self.close_connection()

    def setup_exchange(self, exchange_name):
        if False:
            print('Hello World!')
        'Setup the exchange on RabbitMQ by invoking the Exchange.Declare RPC\n        command. When it is complete, the on_exchange_declareok method will\n        be invoked by pika.\n\n        :param str|unicode exchange_name: The name of the exchange to declare\n\n        '
        LOGGER.info('Declaring exchange: %s', exchange_name)
        cb = functools.partial(self.on_exchange_declareok, userdata=exchange_name)
        self._channel.exchange_declare(exchange=exchange_name, exchange_type=self.EXCHANGE_TYPE, callback=cb)

    def on_exchange_declareok(self, _unused_frame, userdata):
        if False:
            return 10
        'Invoked by pika when RabbitMQ has finished the Exchange.Declare RPC\n        command.\n\n        :param pika.Frame.Method unused_frame: Exchange.DeclareOk response frame\n        :param str|unicode userdata: Extra user data (exchange name)\n\n        '
        LOGGER.info('Exchange declared: %s', userdata)
        self.setup_queue(self.QUEUE)

    def setup_queue(self, queue_name):
        if False:
            while True:
                i = 10
        'Setup the queue on RabbitMQ by invoking the Queue.Declare RPC\n        command. When it is complete, the on_queue_declareok method will\n        be invoked by pika.\n\n        :param str|unicode queue_name: The name of the queue to declare.\n\n        '
        LOGGER.info('Declaring queue %s', queue_name)
        cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
        self._channel.queue_declare(queue=queue_name, callback=cb)

    def on_queue_declareok(self, _unused_frame, userdata):
        if False:
            i = 10
            return i + 15
        'Method invoked by pika when the Queue.Declare RPC call made in\n        setup_queue has completed. In this method we will bind the queue\n        and exchange together with the routing key by issuing the Queue.Bind\n        RPC command. When this command is complete, the on_bindok method will\n        be invoked by pika.\n\n        :param pika.frame.Method _unused_frame: The Queue.DeclareOk frame\n        :param str|unicode userdata: Extra user data (queue name)\n\n        '
        queue_name = userdata
        LOGGER.info('Binding %s to %s with %s', self.EXCHANGE, queue_name, self.ROUTING_KEY)
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        self._channel.queue_bind(queue_name, self.EXCHANGE, routing_key=self.ROUTING_KEY, callback=cb)

    def on_bindok(self, _unused_frame, userdata):
        if False:
            i = 10
            return i + 15
        'Invoked by pika when the Queue.Bind method has completed. At this\n        point we will set the prefetch count for the channel.\n\n        :param pika.frame.Method _unused_frame: The Queue.BindOk response frame\n        :param str|unicode userdata: Extra user data (queue name)\n\n        '
        LOGGER.info('Queue bound: %s', userdata)
        self.set_qos()

    def set_qos(self):
        if False:
            return 10
        'This method sets up the consumer prefetch to only be delivered\n        one message at a time. The consumer must acknowledge this message\n        before RabbitMQ will deliver another one. You should experiment\n        with different prefetch values to achieve desired performance.\n\n        '
        self._channel.basic_qos(prefetch_count=self._prefetch_count, callback=self.on_basic_qos_ok)

    def on_basic_qos_ok(self, _unused_frame):
        if False:
            i = 10
            return i + 15
        'Invoked by pika when the Basic.QoS method has completed. At this\n        point we will start consuming messages by calling start_consuming\n        which will invoke the needed RPC commands to start the process.\n\n        :param pika.frame.Method _unused_frame: The Basic.QosOk response frame\n\n        '
        LOGGER.info('QOS set to: %d', self._prefetch_count)
        self.start_consuming()

    def start_consuming(self):
        if False:
            for i in range(10):
                print('nop')
        'This method sets up the consumer by first calling\n        add_on_cancel_callback so that the object is notified if RabbitMQ\n        cancels the consumer. It then issues the Basic.Consume RPC command\n        which returns the consumer tag that is used to uniquely identify the\n        consumer with RabbitMQ. We keep the value to use it when we want to\n        cancel consuming. The on_message method is passed in as a callback pika\n        will invoke when a message is fully received.\n\n        '
        LOGGER.info('Issuing consumer related RPC commands')
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(self.QUEUE, self.on_message)
        self.was_consuming = True
        self._consuming = True

    def add_on_cancel_callback(self):
        if False:
            return 10
        'Add a callback that will be invoked if RabbitMQ cancels the consumer\n        for some reason. If RabbitMQ does cancel the consumer,\n        on_consumer_cancelled will be invoked by pika.\n\n        '
        LOGGER.info('Adding consumer cancellation callback')
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        if False:
            while True:
                i = 10
        'Invoked by pika when RabbitMQ sends a Basic.Cancel for a consumer\n        receiving messages.\n\n        :param pika.frame.Method method_frame: The Basic.Cancel frame\n\n        '
        LOGGER.info('Consumer was cancelled remotely, shutting down: %r', method_frame)
        if self._channel:
            self._channel.close()

    def on_message(self, _unused_channel, basic_deliver, properties, body):
        if False:
            while True:
                i = 10
        'Invoked by pika when a message is delivered from RabbitMQ. The\n        channel is passed for your convenience. The basic_deliver object that\n        is passed in carries the exchange, routing key, delivery tag and\n        a redelivered flag for the message. The properties passed in is an\n        instance of BasicProperties with the message properties and the body\n        is the message that was sent.\n\n        :param pika.channel.Channel _unused_channel: The channel object\n        :param pika.Spec.Basic.Deliver: basic_deliver method\n        :param pika.Spec.BasicProperties: properties\n        :param bytes body: The message body\n\n        '
        LOGGER.info('Received message # %s from %s: %s', basic_deliver.delivery_tag, properties.app_id, body)
        self.acknowledge_message(basic_deliver.delivery_tag)

    def acknowledge_message(self, delivery_tag):
        if False:
            print('Hello World!')
        'Acknowledge the message delivery from RabbitMQ by sending a\n        Basic.Ack RPC method for the delivery tag.\n\n        :param int delivery_tag: The delivery tag from the Basic.Deliver frame\n\n        '
        LOGGER.info('Acknowledging message %s', delivery_tag)
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        if False:
            while True:
                i = 10
        'Tell RabbitMQ that you would like to stop consuming by sending the\n        Basic.Cancel RPC command.\n\n        '
        if self._channel:
            LOGGER.info('Sending a Basic.Cancel RPC command to RabbitMQ')
            cb = functools.partial(self.on_cancelok, userdata=self._consumer_tag)
            self._channel.basic_cancel(self._consumer_tag, cb)

    def on_cancelok(self, _unused_frame, userdata):
        if False:
            while True:
                i = 10
        'This method is invoked by pika when RabbitMQ acknowledges the\n        cancellation of a consumer. At this point we will close the channel.\n        This will invoke the on_channel_closed method once the channel has been\n        closed, which will in-turn close the connection.\n\n        :param pika.frame.Method _unused_frame: The Basic.CancelOk frame\n        :param str|unicode userdata: Extra user data (consumer tag)\n\n        '
        self._consuming = False
        LOGGER.info('RabbitMQ acknowledged the cancellation of the consumer: %s', userdata)
        self.close_channel()

    def close_channel(self):
        if False:
            while True:
                i = 10
        'Call to close the channel with RabbitMQ cleanly by issuing the\n        Channel.Close RPC command.\n\n        '
        LOGGER.info('Closing the channel')
        self._channel.close()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        'Run the example consumer by connecting to RabbitMQ and then\n        starting the IOLoop to block and allow the AsyncioConnection to operate.\n\n        '
        self._connection = self.connect()
        self._connection.ioloop.run_forever()

    def stop(self):
        if False:
            print('Hello World!')
        'Cleanly shutdown the connection to RabbitMQ by stopping the consumer\n        with RabbitMQ. When RabbitMQ confirms the cancellation, on_cancelok\n        will be invoked by pika, which will then closing the channel and\n        connection. The IOLoop is started again because this method is invoked\n        when CTRL-C is pressed raising a KeyboardInterrupt exception. This\n        exception stops the IOLoop which needs to be running for pika to\n        communicate with RabbitMQ. All of the commands issued prior to starting\n        the IOLoop will be buffered but not processed.\n\n        '
        if not self._closing:
            self._closing = True
            LOGGER.info('Stopping')
            if self._consuming:
                self.stop_consuming()
                self._connection.ioloop.run_forever()
            else:
                self._connection.ioloop.stop()
            LOGGER.info('Stopped')

class ReconnectingExampleConsumer(object):
    """This is an example consumer that will reconnect if the nested
    ExampleConsumer indicates that a reconnect is necessary.

    """

    def __init__(self, amqp_url):
        if False:
            return 10
        self._reconnect_delay = 0
        self._amqp_url = amqp_url
        self._consumer = ExampleConsumer(self._amqp_url)

    def run(self):
        if False:
            return 10
        while True:
            try:
                self._consumer.run()
            except KeyboardInterrupt:
                self._consumer.stop()
                break
            self._maybe_reconnect()

    def _maybe_reconnect(self):
        if False:
            while True:
                i = 10
        if self._consumer.should_reconnect:
            self._consumer.stop()
            reconnect_delay = self._get_reconnect_delay()
            LOGGER.info('Reconnecting after %d seconds', reconnect_delay)
            time.sleep(reconnect_delay)
            self._consumer = ExampleConsumer(self._amqp_url)

    def _get_reconnect_delay(self):
        if False:
            for i in range(10):
                print('nop')
        if self._consumer.was_consuming:
            self._reconnect_delay = 0
        else:
            self._reconnect_delay += 1
        if self._reconnect_delay > 30:
            self._reconnect_delay = 30
        return self._reconnect_delay

def main():
    if False:
        while True:
            i = 10
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    amqp_url = 'amqp://guest:guest@localhost:5672/%2F'
    consumer = ReconnectingExampleConsumer(amqp_url)
    consumer.run()
if __name__ == '__main__':
    main()