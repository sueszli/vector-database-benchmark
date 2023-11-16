import functools
import logging
import json
import pika
from pika.exchange_type import ExchangeType
LOG_FORMAT = '%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s'
LOGGER = logging.getLogger(__name__)

class ExamplePublisher(object):
    """This is an example publisher that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, it will reopen it. You should
    look at the output, as there are limited reasons why the connection may
    be closed, which usually are tied to permission related issues or
    socket timeouts.

    It uses delivery confirmations and illustrates one way to keep track of
    messages that have been sent and if they've been confirmed by RabbitMQ.

    """
    EXCHANGE = 'message'
    EXCHANGE_TYPE = ExchangeType.topic
    PUBLISH_INTERVAL = 1
    QUEUE = 'text'
    ROUTING_KEY = 'example.text'

    def __init__(self, amqp_url):
        if False:
            i = 10
            return i + 15
        'Setup the example publisher object, passing in the URL we will use\n        to connect to RabbitMQ.\n\n        :param str amqp_url: The URL for connecting to RabbitMQ\n\n        '
        self._connection = None
        self._channel = None
        self._deliveries = None
        self._acked = None
        self._nacked = None
        self._message_number = None
        self._stopping = False
        self._url = amqp_url

    def connect(self):
        if False:
            print('Hello World!')
        'This method connects to RabbitMQ, returning the connection handle.\n        When the connection is established, the on_connection_open method\n        will be invoked by pika.\n\n        :rtype: pika.SelectConnection\n\n        '
        LOGGER.info('Connecting to %s', self._url)
        return pika.SelectConnection(pika.URLParameters(self._url), on_open_callback=self.on_connection_open, on_open_error_callback=self.on_connection_open_error, on_close_callback=self.on_connection_closed)

    def on_connection_open(self, _unused_connection):
        if False:
            print('Hello World!')
        "This method is called by pika once the connection to RabbitMQ has\n        been established. It passes the handle to the connection object in\n        case we need it, but in this case, we'll just mark it unused.\n\n        :param pika.SelectConnection _unused_connection: The connection\n\n        "
        LOGGER.info('Connection opened')
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        if False:
            return 10
        "This method is called by pika if the connection to RabbitMQ\n        can't be established.\n\n        :param pika.SelectConnection _unused_connection: The connection\n        :param Exception err: The error\n\n        "
        LOGGER.error('Connection open failed, reopening in 5 seconds: %s', err)
        self._connection.ioloop.call_later(5, self._connection.ioloop.stop)

    def on_connection_closed(self, _unused_connection, reason):
        if False:
            print('Hello World!')
        'This method is invoked by pika when the connection to RabbitMQ is\n        closed unexpectedly. Since it is unexpected, we will reconnect to\n        RabbitMQ if it disconnects.\n\n        :param pika.connection.Connection connection: The closed connection obj\n        :param Exception reason: exception representing reason for loss of\n            connection.\n\n        '
        self._channel = None
        if self._stopping:
            self._connection.ioloop.stop()
        else:
            LOGGER.warning('Connection closed, reopening in 5 seconds: %s', reason)
            self._connection.ioloop.call_later(5, self._connection.ioloop.stop)

    def open_channel(self):
        if False:
            return 10
        'This method will open a new channel with RabbitMQ by issuing the\n        Channel.Open RPC command. When RabbitMQ confirms the channel is open\n        by sending the Channel.OpenOK RPC reply, the on_channel_open method\n        will be invoked.\n\n        '
        LOGGER.info('Creating a new channel')
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        if False:
            i = 10
            return i + 15
        "This method is invoked by pika when the channel has been opened.\n        The channel object is passed in so we can make use of it.\n\n        Since the channel is now open, we'll declare the exchange to use.\n\n        :param pika.channel.Channel channel: The channel object\n\n        "
        LOGGER.info('Channel opened')
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.EXCHANGE)

    def add_on_channel_close_callback(self):
        if False:
            for i in range(10):
                print('nop')
        'This method tells pika to call the on_channel_closed method if\n        RabbitMQ unexpectedly closes the channel.\n\n        '
        LOGGER.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        if False:
            for i in range(10):
                print('nop')
        "Invoked by pika when RabbitMQ unexpectedly closes the channel.\n        Channels are usually closed if you attempt to do something that\n        violates the protocol, such as re-declare an exchange or queue with\n        different parameters. In this case, we'll close the connection\n        to shutdown the object.\n\n        :param pika.channel.Channel channel: The closed channel\n        :param Exception reason: why the channel was closed\n\n        "
        LOGGER.warning('Channel %i was closed: %s', channel, reason)
        self._channel = None
        if not self._stopping:
            self._connection.close()

    def setup_exchange(self, exchange_name):
        if False:
            while True:
                i = 10
        'Setup the exchange on RabbitMQ by invoking the Exchange.Declare RPC\n        command. When it is complete, the on_exchange_declareok method will\n        be invoked by pika.\n\n        :param str|unicode exchange_name: The name of the exchange to declare\n\n        '
        LOGGER.info('Declaring exchange %s', exchange_name)
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
            return 10
        'Setup the queue on RabbitMQ by invoking the Queue.Declare RPC\n        command. When it is complete, the on_queue_declareok method will\n        be invoked by pika.\n\n        :param str|unicode queue_name: The name of the queue to declare.\n\n        '
        LOGGER.info('Declaring queue %s', queue_name)
        self._channel.queue_declare(queue=queue_name, callback=self.on_queue_declareok)

    def on_queue_declareok(self, _unused_frame):
        if False:
            while True:
                i = 10
        'Method invoked by pika when the Queue.Declare RPC call made in\n        setup_queue has completed. In this method we will bind the queue\n        and exchange together with the routing key by issuing the Queue.Bind\n        RPC command. When this command is complete, the on_bindok method will\n        be invoked by pika.\n\n        :param pika.frame.Method method_frame: The Queue.DeclareOk frame\n\n        '
        LOGGER.info('Binding %s to %s with %s', self.EXCHANGE, self.QUEUE, self.ROUTING_KEY)
        self._channel.queue_bind(self.QUEUE, self.EXCHANGE, routing_key=self.ROUTING_KEY, callback=self.on_bindok)

    def on_bindok(self, _unused_frame):
        if False:
            return 10
        "This method is invoked by pika when it receives the Queue.BindOk\n        response from RabbitMQ. Since we know we're now setup and bound, it's\n        time to start publishing."
        LOGGER.info('Queue bound')
        self.start_publishing()

    def start_publishing(self):
        if False:
            i = 10
            return i + 15
        'This method will enable delivery confirmations and schedule the\n        first message to be sent to RabbitMQ\n\n        '
        LOGGER.info('Issuing consumer related RPC commands')
        self.enable_delivery_confirmations()
        self.schedule_next_message()

    def enable_delivery_confirmations(self):
        if False:
            i = 10
            return i + 15
        'Send the Confirm.Select RPC method to RabbitMQ to enable delivery\n        confirmations on the channel. The only way to turn this off is to close\n        the channel and create a new one.\n\n        When the message is confirmed from RabbitMQ, the\n        on_delivery_confirmation method will be invoked passing in a Basic.Ack\n        or Basic.Nack method from RabbitMQ that will indicate which messages it\n        is confirming or rejecting.\n\n        '
        LOGGER.info('Issuing Confirm.Select RPC command')
        self._channel.confirm_delivery(self.on_delivery_confirmation)

    def on_delivery_confirmation(self, method_frame):
        if False:
            i = 10
            return i + 15
        "Invoked by pika when RabbitMQ responds to a Basic.Publish RPC\n        command, passing in either a Basic.Ack or Basic.Nack frame with\n        the delivery tag of the message that was published. The delivery tag\n        is an integer counter indicating the message number that was sent\n        on the channel via Basic.Publish. Here we're just doing house keeping\n        to keep track of stats and remove message numbers that we expect\n        a delivery confirmation of from the list used to keep track of messages\n        that are pending confirmation.\n\n        :param pika.frame.Method method_frame: Basic.Ack or Basic.Nack frame\n\n        "
        confirmation_type = method_frame.method.NAME.split('.')[1].lower()
        ack_multiple = method_frame.method.multiple
        delivery_tag = method_frame.method.delivery_tag
        LOGGER.info('Received %s for delivery tag: %i (multiple: %s)', confirmation_type, delivery_tag, ack_multiple)
        if confirmation_type == 'ack':
            self._acked += 1
        elif confirmation_type == 'nack':
            self._nacked += 1
        del self._deliveries[delivery_tag]
        if ack_multiple:
            for tmp_tag in list(self._deliveries.keys()):
                if tmp_tag <= delivery_tag:
                    self._acked += 1
                    del self._deliveries[tmp_tag]
        '\n        NOTE: at some point you would check self._deliveries for stale\n        entries and decide to attempt re-delivery\n        '
        LOGGER.info('Published %i messages, %i have yet to be confirmed, %i were acked and %i were nacked', self._message_number, len(self._deliveries), self._acked, self._nacked)

    def schedule_next_message(self):
        if False:
            print('Hello World!')
        'If we are not closing our connection to RabbitMQ, schedule another\n        message to be delivered in PUBLISH_INTERVAL seconds.\n\n        '
        LOGGER.info('Scheduling next message for %0.1f seconds', self.PUBLISH_INTERVAL)
        self._connection.ioloop.call_later(self.PUBLISH_INTERVAL, self.publish_message)

    def publish_message(self):
        if False:
            print('Hello World!')
        'If the class is not stopping, publish a message to RabbitMQ,\n        appending a list of deliveries with the message number that was sent.\n        This list will be used to check for delivery confirmations in the\n        on_delivery_confirmations method.\n\n        Once the message has been sent, schedule another message to be sent.\n        The main reason I put scheduling in was just so you can get a good idea\n        of how the process is flowing by slowing down and speeding up the\n        delivery intervals by changing the PUBLISH_INTERVAL constant in the\n        class.\n\n        '
        if self._channel is None or not self._channel.is_open:
            return
        hdrs = {u'مفتاح': u' قيمة', u'键': u'值', u'キー': u'値'}
        properties = pika.BasicProperties(app_id='example-publisher', content_type='application/json', headers=hdrs)
        message = u'مفتاح قيمة 键 值 キー 値'
        self._channel.basic_publish(self.EXCHANGE, self.ROUTING_KEY, json.dumps(message, ensure_ascii=False), properties)
        self._message_number += 1
        self._deliveries[self._message_number] = True
        LOGGER.info('Published message # %i', self._message_number)
        self.schedule_next_message()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        'Run the example code by connecting and then starting the IOLoop.\n\n        '
        while not self._stopping:
            self._connection = None
            self._deliveries = {}
            self._acked = 0
            self._nacked = 0
            self._message_number = 0
            try:
                self._connection = self.connect()
                self._connection.ioloop.start()
            except KeyboardInterrupt:
                self.stop()
                if self._connection is not None and (not self._connection.is_closed):
                    self._connection.ioloop.start()
        LOGGER.info('Stopped')

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stop the example by closing the channel and connection. We\n        set a flag here so that we stop scheduling new messages to be\n        published. The IOLoop is started because this method is\n        invoked by the Try/Catch below when KeyboardInterrupt is caught.\n        Starting the IOLoop again will allow the publisher to cleanly\n        disconnect from RabbitMQ.\n\n        '
        LOGGER.info('Stopping')
        self._stopping = True
        self.close_channel()
        self.close_connection()

    def close_channel(self):
        if False:
            i = 10
            return i + 15
        'Invoke this command to close the channel with RabbitMQ by sending\n        the Channel.Close RPC command.\n\n        '
        if self._channel is not None:
            LOGGER.info('Closing the channel')
            self._channel.close()

    def close_connection(self):
        if False:
            while True:
                i = 10
        'This method closes the connection to RabbitMQ.'
        if self._connection is not None:
            LOGGER.info('Closing connection')
            self._connection.close()

def main():
    if False:
        while True:
            i = 10
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    example = ExamplePublisher('amqp://guest:guest@localhost:5672/%2F?connection_attempts=3&heartbeat=3600')
    example.run()
if __name__ == '__main__':
    main()