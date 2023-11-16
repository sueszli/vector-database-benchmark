"""Using Pika with a Twisted reactor.

The interfaces in this module are Deferred-based when possible. This means that
the connection.channel() method and most of the channel methods return
Deferreds instead of taking a callback argument and that basic_consume()
returns a Twisted DeferredQueue where messages from the server will be
stored. Refer to the docstrings for TwistedProtocolConnection.channel() and the
TwistedChannel class for details.

"""
import functools
import logging
from collections import namedtuple
from twisted.internet import defer, error as twisted_error, reactor, protocol
from twisted.python.failure import Failure
import pika.connection
from pika import exceptions, spec
from pika.adapters.utils import nbio_interface
from pika.adapters.utils.io_services_utils import check_callback_arg
from pika.exchange_type import ExchangeType
LOGGER = logging.getLogger(__name__)

class ClosableDeferredQueue(defer.DeferredQueue):
    """
    Like the normal Twisted DeferredQueue, but after close() is called with an
    exception instance all pending Deferreds are errbacked and further attempts
    to call get() or put() return a Failure wrapping that exception.
    """

    def __init__(self, size=None, backlog=None):
        if False:
            while True:
                i = 10
        self.closed = None
        super().__init__(size, backlog)

    def put(self, obj):
        if False:
            while True:
                i = 10
        '\n        Like the original :meth:`DeferredQueue.put` method, but returns an\n        errback if the queue is closed.\n\n        '
        if self.closed:
            LOGGER.error('Impossible to put to the queue, it is closed.')
            return defer.fail(self.closed)
        return defer.DeferredQueue.put(self, obj)

    def get(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a Deferred that will fire with the next item in the queue, when\n        it's available.\n\n        The Deferred will errback if the queue is closed.\n\n        :returns: Deferred that fires with the next item.\n        :rtype: Deferred\n\n        "
        if self.closed:
            LOGGER.error('Impossible to get from the queue, it is closed.')
            return defer.fail(self.closed)
        return defer.DeferredQueue.get(self)

    def close(self, reason):
        if False:
            return 10
        'Closes the queue.\n\n        Errback the pending calls to :meth:`get()`.\n\n        '
        if self.closed:
            LOGGER.warning('Queue was already closed with reason: %s.', self.closed)
        self.closed = reason
        while self.waiting:
            self.waiting.pop().errback(reason)
        self.pending = []
ReceivedMessage = namedtuple('ReceivedMessage', ['channel', 'method', 'properties', 'body'])

class TwistedChannel:
    """A wrapper around Pika's Channel.

    Channel methods that normally take a callback argument are wrapped to
    return a Deferred that fires with whatever would be passed to the callback.
    If the channel gets closed, all pending Deferreds are errbacked with a
    ChannelClosed exception. The returned Deferreds fire with whatever
    arguments the callback to the original method would receive.

    Some methods like basic_consume and basic_get are wrapped in a special way,
    see their docstrings for details.
    """

    def __init__(self, channel):
        if False:
            for i in range(10):
                print('nop')
        self._channel = channel
        self._closed = None
        self._calls = set()
        self._consumers = {}
        self._basic_get_deferred = None
        self._channel.add_callback(self._on_getempty, [spec.Basic.GetEmpty], False)
        self._queue_name_to_consumer_tags = {}
        self._delivery_confirmation = False
        self._delivery_message_id = None
        self._deliveries = {}
        self._puback_return = None
        self.on_closed = defer.Deferred()
        self._channel.add_on_close_callback(self._on_channel_closed)
        self._channel.add_on_cancel_callback(self._on_consumer_cancelled_by_broker)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{cls} channel={chan!r}>'.format(cls=self.__class__.__name__, chan=self._channel)

    def _on_channel_closed(self, _channel, reason):
        if False:
            print('Hello World!')
        self._closed = reason
        for d in self._calls:
            d.errback(self._closed)
        for d in self._deliveries.values():
            d.errback(self._closed)
        for consumer in self._consumers.values():
            consumer.close(self._closed)
        self._calls = set()
        self._deliveries = {}
        self._consumers = {}
        self.on_closed.callback(self._closed)

    def _on_consumer_cancelled_by_broker(self, method_frame):
        if False:
            for i in range(10):
                print('nop')
        'Called by impl when broker cancels consumer via Basic.Cancel.\n\n        This is a RabbitMQ-specific feature. The circumstances include deletion\n        of queue being consumed as well as failure of a HA node responsible for\n        the queue being consumed.\n\n        :param pika.frame.Method method_frame: method frame with the\n            `spec.Basic.Cancel` method\n\n        '
        return self._on_consumer_cancelled(method_frame)

    def _on_consumer_cancelled(self, frame):
        if False:
            while True:
                i = 10
        'Called when the broker cancels a consumer via Basic.Cancel or when\n        the broker responds to a Basic.Cancel request by Basic.CancelOk.\n\n        :param pika.frame.Method frame: method frame with the\n            `spec.Basic.Cancel` or `spec.Basic.CancelOk` method\n\n        '
        consumer_tag = frame.method.consumer_tag
        if consumer_tag not in self._consumers:
            LOGGER.warning('basic_cancel - consumer not found: %s', consumer_tag)
            return frame
        self._consumers[consumer_tag].close(exceptions.ConsumerCancelled())
        del self._consumers[consumer_tag]
        for ctags in self._queue_name_to_consumer_tags.values():
            try:
                ctags.remove(consumer_tag)
            except KeyError:
                continue
        return frame

    def _on_getempty(self, _method_frame):
        if False:
            return 10
        'Callback the Basic.Get deferred with None.\n        '
        if self._basic_get_deferred is None:
            LOGGER.warning('Got Basic.GetEmpty but no Basic.Get calls were pending.')
            return
        self._basic_get_deferred.callback(None)

    def _wrap_channel_method(self, name):
        if False:
            for i in range(10):
                print('nop')
        "Wrap Pika's Channel method to make it return a Deferred that fires\n        when the method completes and errbacks if the channel gets closed. If\n        the original method's callback would receive more than one argument,\n        the Deferred fires with a tuple of argument values.\n\n        "
        method = getattr(self._channel, name)

        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            if False:
                return 10
            if self._closed:
                return defer.fail(self._closed)
            d = defer.Deferred()
            self._calls.add(d)
            d.addCallback(self._clear_call, d)

            def single_argument(*args):
                if False:
                    return 10
                '\n                Make sure that the deferred is called with a single argument.\n                In case the original callback fires with more than one, convert\n                to a tuple.\n                '
                if len(args) > 1:
                    d.callback(tuple(args))
                else:
                    d.callback(*args)
            kwargs['callback'] = single_argument
            try:
                method(*args, **kwargs)
            except Exception:
                return defer.fail()
            return d
        return wrapped

    def _clear_call(self, ret, d):
        if False:
            for i in range(10):
                print('nop')
        self._calls.discard(d)
        return ret

    @property
    def channel_number(self):
        if False:
            print('Hello World!')
        return self._channel.channel_number

    @property
    def connection(self):
        if False:
            return 10
        return self._channel.connection

    @property
    def is_closed(self):
        if False:
            i = 10
            return i + 15
        'Returns True if the channel is closed.\n\n        :rtype: bool\n\n        '
        return self._channel.is_closed

    @property
    def is_closing(self):
        if False:
            print('Hello World!')
        'Returns True if client-initiated closing of the channel is in\n        progress.\n\n        :rtype: bool\n\n        '
        return self._channel.is_closing

    @property
    def is_open(self):
        if False:
            while True:
                i = 10
        'Returns True if the channel is open.\n\n        :rtype: bool\n\n        '
        return self._channel.is_open

    @property
    def flow_active(self):
        if False:
            while True:
                i = 10
        return self._channel.flow_active

    @property
    def consumer_tags(self):
        if False:
            print('Hello World!')
        return self._channel.consumer_tags

    def callback_deferred(self, deferred, replies):
        if False:
            print('Hello World!')
        "Pass in a Deferred and a list replies from the RabbitMQ broker which\n        you'd like the Deferred to be callbacked on with the frame as callback\n        value.\n\n        :param Deferred deferred: The Deferred to callback\n        :param list replies: The replies to callback on\n\n        "
        self._channel.add_callback(deferred.callback, replies)

    def add_on_return_callback(self, callback):
        if False:
            while True:
                i = 10
        'Pass a callback function that will be called when a published\n        message is rejected and returned by the server via `Basic.Return`.\n\n        :param callable callback: The method to call on callback with the\n            message as only argument. The message is a named tuple with\n            the following attributes\n            - channel: this TwistedChannel\n            - method: pika.spec.Basic.Return\n            - properties: pika.spec.BasicProperties\n            - body: bytes\n        '
        self._channel.add_on_return_callback(lambda _channel, method, properties, body: callback(ReceivedMessage(channel=self, method=method, properties=properties, body=body)))

    def basic_ack(self, delivery_tag=0, multiple=False):
        if False:
            return 10
        'Acknowledge one or more messages. When sent by the client, this\n        method acknowledges one or more messages delivered via the Deliver or\n        Get-Ok methods. When sent by server, this method acknowledges one or\n        more messages published with the Publish method on a channel in\n        confirm mode. The acknowledgement can be for a single message or a\n        set of messages up to and including a specific message.\n\n        :param integer delivery_tag: int/long The server-assigned delivery tag\n        :param bool multiple: If set to True, the delivery tag is treated as\n                              "up to and including", so that multiple messages\n                              can be acknowledged with a single method. If set\n                              to False, the delivery tag refers to a single\n                              message. If the multiple field is 1, and the\n                              delivery tag is zero, this indicates\n                              acknowledgement of all outstanding messages.\n\n        '
        return self._channel.basic_ack(delivery_tag=delivery_tag, multiple=multiple)

    def basic_cancel(self, consumer_tag=''):
        if False:
            return 10
        'This method cancels a consumer. This does not affect already\n        delivered messages, but it does mean the server will not send any more\n        messages for that consumer. The client may receive an arbitrary number\n        of messages in between sending the cancel method and receiving the\n        cancel-ok reply. It may also be sent from the server to the client in\n        the event of the consumer being unexpectedly cancelled (i.e. cancelled\n        for any reason other than the server receiving the corresponding\n        basic.cancel from the client). This allows clients to be notified of\n        the loss of consumers due to events such as queue deletion.\n\n        This method wraps :meth:`Channel.basic_cancel\n        <pika.channel.Channel.basic_cancel>` and closes any deferred queue\n        associated with that consumer.\n\n        :param str consumer_tag: Identifier for the consumer\n        :returns: Deferred that fires on the Basic.CancelOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        wrapped = self._wrap_channel_method('basic_cancel')
        d = wrapped(consumer_tag=consumer_tag)
        return d.addCallback(self._on_consumer_cancelled)

    def basic_consume(self, queue, auto_ack=False, exclusive=False, consumer_tag=None, arguments=None):
        if False:
            for i in range(10):
                print('nop')
        "Consume from a server queue.\n\n        Sends the AMQP 0-9-1 command Basic.Consume to the broker and binds\n        messages for the consumer_tag to a\n        :class:`ClosableDeferredQueue`. If you do not pass in a\n        consumer_tag, one will be automatically generated for you.\n\n        For more information on basic_consume, see:\n        Tutorial 2 at http://www.rabbitmq.com/getstarted.html\n        http://www.rabbitmq.com/confirms.html\n        http://www.rabbitmq.com/amqp-0-9-1-reference.html#basic.consume\n\n        :param str queue: The queue to consume from. Use the empty string to\n            specify the most recent server-named queue for this channel.\n        :param bool auto_ack: if set to True, automatic acknowledgement mode\n            will be used (see http://www.rabbitmq.com/confirms.html). This\n            corresponds with the 'no_ack' parameter in the basic.consume AMQP\n            0.9.1 method\n        :param bool exclusive: Don't allow other consumers on the queue\n        :param str consumer_tag: Specify your own consumer tag\n        :param dict arguments: Custom key/value pair arguments for the consumer\n        :returns: Deferred that fires with a tuple\n            ``(queue_object, consumer_tag)``. The Deferred will errback with an\n            instance of :class:`exceptions.ChannelClosed` if the call fails.\n            The queue object is an instance of :class:`ClosableDeferredQueue`,\n            where data received from the queue will be stored. Clients should\n            use its :meth:`get() <ClosableDeferredQueue.get>` method to fetch\n            an individual message, which will return a Deferred firing with a\n            namedtuple whose attributes are:\n            - channel: this TwistedChannel\n            - method: pika.spec.Basic.Deliver\n            - properties: pika.spec.BasicProperties\n            - body: bytes\n        :rtype: Deferred\n\n        "
        if self._closed:
            return defer.fail(self._closed)
        queue_obj = ClosableDeferredQueue()
        d = defer.Deferred()
        self._calls.add(d)

        def on_consume_ok(frame):
            if False:
                return 10
            consumer_tag = frame.method.consumer_tag
            self._queue_name_to_consumer_tags.setdefault(queue, set()).add(consumer_tag)
            self._consumers[consumer_tag] = queue_obj
            self._calls.discard(d)
            d.callback((queue_obj, consumer_tag))

        def on_message_callback(_channel, method, properties, body):
            if False:
                for i in range(10):
                    print('nop')
            'Add the ReceivedMessage to the queue, while replacing the\n            channel implementation.\n            '
            queue_obj.put(ReceivedMessage(channel=self, method=method, properties=properties, body=body))
        try:
            self._channel.basic_consume(queue=queue, on_message_callback=on_message_callback, auto_ack=auto_ack, exclusive=exclusive, consumer_tag=consumer_tag, arguments=arguments, callback=on_consume_ok)
        except Exception:
            return defer.fail()
        return d

    def basic_get(self, queue, auto_ack=False):
        if False:
            print('Hello World!')
        'Get a single message from the AMQP broker.\n\n        Will return If the queue is empty, it will return None.\n        If you want to\n        be notified of Basic.GetEmpty, use the Channel.add_callback method\n        adding your Basic.GetEmpty callback which should expect only one\n        parameter, frame. Due to implementation details, this cannot be called\n        a second time until the callback is executed.  For more information on\n        basic_get and its parameters, see:\n\n        http://www.rabbitmq.com/amqp-0-9-1-reference.html#basic.get\n\n        This method wraps :meth:`Channel.basic_get\n        <pika.channel.Channel.basic_get>`.\n\n        :param str queue: The queue from which to get a message. Use the empty\n                      string to specify the most recent server-named queue\n                      for this channel.\n        :param bool auto_ack: Tell the broker to not expect a reply\n        :returns: Deferred that fires with a namedtuple whose attributes are:\n             - channel: this TwistedChannel\n             - method: pika.spec.Basic.GetOk\n             - properties: pika.spec.BasicProperties\n             - body: bytes\n            If the queue is empty, None will be returned.\n        :rtype: Deferred\n        :raises pika.exceptions.DuplicateGetOkCallback:\n\n        '
        if self._basic_get_deferred is not None:
            raise exceptions.DuplicateGetOkCallback()

        def create_namedtuple(result):
            if False:
                print('Hello World!')
            if result is None:
                return None
            (_channel, method, properties, body) = result
            return ReceivedMessage(channel=self, method=method, properties=properties, body=body)

        def cleanup_attribute(result):
            if False:
                return 10
            self._basic_get_deferred = None
            return result
        d = self._wrap_channel_method('basic_get')(queue=queue, auto_ack=auto_ack)
        d.addCallback(create_namedtuple)
        d.addBoth(cleanup_attribute)
        self._basic_get_deferred = d
        return d

    def basic_nack(self, delivery_tag=None, multiple=False, requeue=True):
        if False:
            while True:
                i = 10
        'This method allows a client to reject one or more incoming messages.\n        It can be used to interrupt and cancel large incoming messages, or\n        return untreatable messages to their original queue.\n\n        :param integer delivery_tag: int/long The server-assigned delivery tag\n        :param bool multiple: If set to True, the delivery tag is treated as\n                              "up to and including", so that multiple messages\n                              can be acknowledged with a single method. If set\n                              to False, the delivery tag refers to a single\n                              message. If the multiple field is 1, and the\n                              delivery tag is zero, this indicates\n                              acknowledgement of all outstanding messages.\n        :param bool requeue: If requeue is true, the server will attempt to\n                             requeue the message. If requeue is false or the\n                             requeue attempt fails the messages are discarded\n                             or dead-lettered.\n\n        '
        return self._channel.basic_nack(delivery_tag=delivery_tag, multiple=multiple, requeue=requeue)

    def basic_publish(self, exchange, routing_key, body, properties=None, mandatory=False):
        if False:
            i = 10
            return i + 15
        "Publish to the channel with the given exchange, routing key and body.\n\n        This method wraps :meth:`Channel.basic_publish\n        <pika.channel.Channel.basic_publish>`, but makes sure the channel is\n        not closed before publishing.\n\n        For more information on basic_publish and what the parameters do, see:\n\n        http://www.rabbitmq.com/amqp-0-9-1-reference.html#basic.publish\n\n        :param str exchange: The exchange to publish to\n        :param str routing_key: The routing key to bind on\n        :param bytes body: The message body\n        :param pika.spec.BasicProperties properties: Basic.properties\n        :param bool mandatory: The mandatory flag\n        :returns: A Deferred that fires with the result of the channel's\n            basic_publish.\n        :rtype: Deferred\n        :raises UnroutableError: raised when a message published in\n            publisher-acknowledgments mode (see\n            `BlockingChannel.confirm_delivery`) is returned via `Basic.Return`\n            followed by `Basic.Ack`.\n        :raises NackError: raised when a message published in\n            publisher-acknowledgements mode is Nack'ed by the broker. See\n            `BlockingChannel.confirm_delivery`.\n\n        "
        if self._closed:
            return defer.fail(self._closed)
        result = self._channel.basic_publish(exchange=exchange, routing_key=routing_key, body=body, properties=properties, mandatory=mandatory)
        if not self._delivery_confirmation:
            return defer.succeed(result)
        else:
            self._delivery_message_id += 1
            self._deliveries[self._delivery_message_id] = defer.Deferred()
            return self._deliveries[self._delivery_message_id]

    def basic_qos(self, prefetch_size=0, prefetch_count=0, global_qos=False):
        if False:
            i = 10
            return i + 15
        'Specify quality of service. This method requests a specific quality\n        of service. The QoS can be specified for the current channel or for all\n        channels on the connection. The client can request that messages be\n        sent in advance so that when the client finishes processing a message,\n        the following message is already held locally, rather than needing to\n        be sent down the channel. Prefetching gives a performance improvement.\n\n        :param int prefetch_size:  This field specifies the prefetch window\n                                   size. The server will send a message in\n                                   advance if it is equal to or smaller in size\n                                   than the available prefetch size (and also\n                                   falls into other prefetch limits). May be\n                                   set to zero, meaning "no specific limit",\n                                   although other prefetch limits may still\n                                   apply. The prefetch-size is ignored by\n                                   consumers who have enabled the no-ack\n                                   option.\n        :param int prefetch_count: Specifies a prefetch window in terms of\n                                   whole messages. This field may be used in\n                                   combination with the prefetch-size field; a\n                                   message will only be sent in advance if both\n                                   prefetch windows (and those at the channel\n                                   and connection level) allow it. The\n                                   prefetch-count is ignored by consumers who\n                                   have enabled the no-ack option.\n        :param bool global_qos:    Should the QoS apply to all channels on the\n                                   connection.\n        :returns: Deferred that fires on the Basic.QosOk response\n        :rtype: Deferred\n\n        '
        return self._wrap_channel_method('basic_qos')(prefetch_size=prefetch_size, prefetch_count=prefetch_count, global_qos=global_qos)

    def basic_reject(self, delivery_tag, requeue=True):
        if False:
            for i in range(10):
                print('nop')
        'Reject an incoming message. This method allows a client to reject a\n        message. It can be used to interrupt and cancel large incoming\n        messages, or return untreatable messages to their original queue.\n\n        :param integer delivery_tag: int/long The server-assigned delivery tag\n        :param bool requeue: If requeue is true, the server will attempt to\n                             requeue the message. If requeue is false or the\n                             requeue attempt fails the messages are discarded\n                             or dead-lettered.\n        :raises: TypeError\n\n        '
        return self._channel.basic_reject(delivery_tag=delivery_tag, requeue=requeue)

    def basic_recover(self, requeue=False):
        if False:
            for i in range(10):
                print('nop')
        'This method asks the server to redeliver all unacknowledged messages\n        on a specified channel. Zero or more messages may be redelivered. This\n        method replaces the asynchronous Recover.\n\n        :param bool requeue: If False, the message will be redelivered to the\n                             original recipient. If True, the server will\n                             attempt to requeue the message, potentially then\n                             delivering it to an alternative subscriber.\n        :returns: Deferred that fires on the Basic.RecoverOk response\n        :rtype: Deferred\n\n        '
        return self._wrap_channel_method('basic_recover')(requeue=requeue)

    def close(self, reply_code=0, reply_text='Normal shutdown'):
        if False:
            return 10
        'Invoke a graceful shutdown of the channel with the AMQP Broker.\n\n        If channel is OPENING, transition to CLOSING and suppress the incoming\n        Channel.OpenOk, if any.\n\n        :param int reply_code: The reason code to send to broker\n        :param str reply_text: The reason text to send to broker\n\n        :raises ChannelWrongStateError: if channel is closed or closing\n\n        '
        return self._channel.close(reply_code=reply_code, reply_text=reply_text)

    def confirm_delivery(self):
        if False:
            return 10
        'Turn on Confirm mode in the channel. Pass in a callback to be\n        notified by the Broker when a message has been confirmed as received or\n        rejected (Basic.Ack, Basic.Nack) from the broker to the publisher.\n\n        For more information see:\n            http://www.rabbitmq.com/confirms.html#publisher-confirms\n\n        :returns: Deferred that fires on the Confirm.SelectOk response\n        :rtype: Deferred\n\n        '
        if self._delivery_confirmation:
            LOGGER.error('confirm_delivery: confirmation was already enabled.')
            return defer.succeed(None)
        wrapped = self._wrap_channel_method('confirm_delivery')
        d = wrapped(ack_nack_callback=self._on_delivery_confirmation)

        def set_delivery_confirmation(result):
            if False:
                print('Hello World!')
            self._delivery_confirmation = True
            self._delivery_message_id = 0
            LOGGER.debug('Delivery confirmation enabled.')
            return result
        d.addCallback(set_delivery_confirmation)
        self._channel.add_on_return_callback(self._on_puback_message_returned)
        return d

    def _on_delivery_confirmation(self, method_frame):
        if False:
            for i in range(10):
                print('nop')
        "Invoked by pika when RabbitMQ responds to a Basic.Publish RPC\n        command, passing in either a Basic.Ack or Basic.Nack frame with\n        the delivery tag of the message that was published. The delivery tag\n        is an integer counter indicating the message number that was sent\n        on the channel via Basic.Publish. Here we're just doing house keeping\n        to keep track of stats and remove message numbers that we expect\n        a delivery confirmation of from the list used to keep track of messages\n        that are pending confirmation.\n\n        :param pika.frame.Method method_frame: Basic.Ack or Basic.Nack frame\n\n        "
        delivery_tag = method_frame.method.delivery_tag
        if delivery_tag not in self._deliveries:
            LOGGER.error('Delivery tag %s not found in the pending deliveries', delivery_tag)
            return
        if method_frame.method.multiple:
            tags = [tag for tag in self._deliveries if tag <= delivery_tag]
            tags.sort()
        else:
            tags = [delivery_tag]
        for tag in tags:
            d = self._deliveries[tag]
            del self._deliveries[tag]
            if isinstance(method_frame.method, pika.spec.Basic.Nack):
                LOGGER.warning("Message was Nack'ed by broker: nack=%r; channel=%s;", method_frame.method, self.channel_number)
                if self._puback_return is not None:
                    returned_messages = [self._puback_return]
                    self._puback_return = None
                else:
                    returned_messages = []
                d.errback(exceptions.NackError(returned_messages))
            else:
                assert isinstance(method_frame.method, pika.spec.Basic.Ack)
                if self._puback_return is not None:
                    returned_messages = [self._puback_return]
                    self._puback_return = None
                    d.errback(exceptions.UnroutableError(returned_messages))
                else:
                    d.callback(method_frame.method)

    def _on_puback_message_returned(self, channel, method, properties, body):
        if False:
            return 10
        'Called as the result of Basic.Return from broker in\n        publisher-acknowledgements mode. Saves the info as a ReturnedMessage\n        instance in self._puback_return.\n\n        :param pika.Channel channel: our self._impl channel\n        :param pika.spec.Basic.Return method:\n        :param pika.spec.BasicProperties properties: message properties\n        :param bytes body: returned message body; empty string if no body\n\n        '
        assert isinstance(method, spec.Basic.Return), method
        assert isinstance(properties, spec.BasicProperties), properties
        LOGGER.warning('Published message was returned: _delivery_confirmation=%s; channel=%s; method=%r; properties=%r; body_size=%d; body_prefix=%.255r', self._delivery_confirmation, channel.channel_number, method, properties, len(body) if body is not None else None, body)
        self._puback_return = ReceivedMessage(channel=self, method=method, properties=properties, body=body)

    def exchange_bind(self, destination, source, routing_key='', arguments=None):
        if False:
            return 10
        'Bind an exchange to another exchange.\n\n        :param str destination: The destination exchange to bind\n        :param str source: The source exchange to bind to\n        :param str routing_key: The routing key to bind on\n        :param dict arguments: Custom key/value pair arguments for the binding\n        :raises ValueError:\n        :returns: Deferred that fires on the Exchange.BindOk response\n        :rtype: Deferred\n\n        '
        return self._wrap_channel_method('exchange_bind')(destination=destination, source=source, routing_key=routing_key, arguments=arguments)

    def exchange_declare(self, exchange, exchange_type=ExchangeType.direct, passive=False, durable=False, auto_delete=False, internal=False, arguments=None):
        if False:
            for i in range(10):
                print('nop')
        'This method creates an exchange if it does not already exist, and if\n        the exchange exists, verifies that it is of the correct and expected\n        class.\n\n        If passive set, the server will reply with Declare-Ok if the exchange\n        already exists with the same name, and raise an error if not and if the\n        exchange does not already exist, the server MUST raise a channel\n        exception with reply code 404 (not found).\n\n        :param str exchange: The exchange name consists of a non-empty sequence\n            of these characters: letters, digits, hyphen, underscore, period,\n            or colon\n        :param str exchange_type: The exchange type to use\n        :param bool passive: Perform a declare or just check to see if it\n            exists\n        :param bool durable: Survive a reboot of RabbitMQ\n        :param bool auto_delete: Remove when no more queues are bound to it\n        :param bool internal: Can only be published to by other exchanges\n        :param dict arguments: Custom key/value pair arguments for the exchange\n        :returns: Deferred that fires on the Exchange.DeclareOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('exchange_declare')(exchange=exchange, exchange_type=exchange_type, passive=passive, durable=durable, auto_delete=auto_delete, internal=internal, arguments=arguments)

    def exchange_delete(self, exchange=None, if_unused=False):
        if False:
            print('Hello World!')
        'Delete the exchange.\n\n        :param str exchange: The exchange name\n        :param bool if_unused: only delete if the exchange is unused\n        :returns: Deferred that fires on the Exchange.DeleteOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('exchange_delete')(exchange=exchange, if_unused=if_unused)

    def exchange_unbind(self, destination=None, source=None, routing_key='', arguments=None):
        if False:
            while True:
                i = 10
        'Unbind an exchange from another exchange.\n\n        :param str destination: The destination exchange to unbind\n        :param str source: The source exchange to unbind from\n        :param str routing_key: The routing key to unbind\n        :param dict arguments: Custom key/value pair arguments for the binding\n        :returns: Deferred that fires on the Exchange.UnbindOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('exchange_unbind')(destination=destination, source=source, routing_key=routing_key, arguments=arguments)

    def flow(self, active):
        if False:
            while True:
                i = 10
        'Turn Channel flow control off and on.\n\n        Returns a Deferred that will fire with a bool indicating the channel\n        flow state. For more information, please reference:\n\n        http://www.rabbitmq.com/amqp-0-9-1-reference.html#channel.flow\n\n        :param bool active: Turn flow on or off\n        :returns: Deferred that fires with the channel flow state\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('flow')(active=active)

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        'Open the channel'
        return self._channel.open()

    def queue_bind(self, queue, exchange, routing_key=None, arguments=None):
        if False:
            for i in range(10):
                print('nop')
        'Bind the queue to the specified exchange\n\n        :param str queue: The queue to bind to the exchange\n        :param str exchange: The source exchange to bind to\n        :param str routing_key: The routing key to bind on\n        :param dict arguments: Custom key/value pair arguments for the binding\n        :returns: Deferred that fires on the Queue.BindOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('queue_bind')(queue=queue, exchange=exchange, routing_key=routing_key, arguments=arguments)

    def queue_declare(self, queue, passive=False, durable=False, exclusive=False, auto_delete=False, arguments=None):
        if False:
            while True:
                i = 10
        'Declare queue, create if needed. This method creates or checks a\n        queue. When creating a new queue the client can specify various\n        properties that control the durability of the queue and its contents,\n        and the level of sharing for the queue.\n\n        Use an empty string as the queue name for the broker to auto-generate\n        one\n\n        :param str queue: The queue name; if empty string, the broker will\n            create a unique queue name\n        :param bool passive: Only check to see if the queue exists\n        :param bool durable: Survive reboots of the broker\n        :param bool exclusive: Only allow access by the current connection\n        :param bool auto_delete: Delete after consumer cancels or disconnects\n        :param dict arguments: Custom key/value arguments for the queue\n        :returns: Deferred that fires on the Queue.DeclareOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('queue_declare')(queue=queue, passive=passive, durable=durable, exclusive=exclusive, auto_delete=auto_delete, arguments=arguments)

    def queue_delete(self, queue, if_unused=False, if_empty=False):
        if False:
            print('Hello World!')
        "Delete a queue from the broker.\n\n\n        This method wraps :meth:`Channel.queue_delete\n        <pika.channel.Channel.queue_delete>`, and removes the reference to the\n        queue object after it gets deleted on the server.\n\n        :param str queue: The queue to delete\n        :param bool if_unused: only delete if it's unused\n        :param bool if_empty: only delete if the queue is empty\n        :returns: Deferred that fires on the Queue.DeleteOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        "
        wrapped = self._wrap_channel_method('queue_delete')
        d = wrapped(queue=queue, if_unused=if_unused, if_empty=if_empty)

        def _clear_consumer(ret, queue_name):
            if False:
                i = 10
                return i + 15
            for consumer_tag in list(self._queue_name_to_consumer_tags.get(queue_name, set())):
                self._consumers[consumer_tag].close(exceptions.ConsumerCancelled('Queue %s was deleted.' % queue_name))
                del self._consumers[consumer_tag]
                self._queue_name_to_consumer_tags[queue_name].remove(consumer_tag)
            return ret
        return d.addCallback(_clear_consumer, queue)

    def queue_purge(self, queue):
        if False:
            print('Hello World!')
        'Purge all of the messages from the specified queue\n\n        :param str queue: The queue to purge\n        :returns: Deferred that fires on the Queue.PurgeOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('queue_purge')(queue=queue)

    def queue_unbind(self, queue, exchange=None, routing_key=None, arguments=None):
        if False:
            print('Hello World!')
        'Unbind a queue from an exchange.\n\n        :param str queue: The queue to unbind from the exchange\n        :param str exchange: The source exchange to bind from\n        :param str routing_key: The routing key to unbind\n        :param dict arguments: Custom key/value pair arguments for the binding\n        :returns: Deferred that fires on the Queue.UnbindOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('queue_unbind')(queue=queue, exchange=exchange, routing_key=routing_key, arguments=arguments)

    def tx_commit(self):
        if False:
            i = 10
            return i + 15
        'Commit a transaction.\n\n        :returns: Deferred that fires on the Tx.CommitOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('tx_commit')()

    def tx_rollback(self):
        if False:
            i = 10
            return i + 15
        'Rollback a transaction.\n\n        :returns: Deferred that fires on the Tx.RollbackOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('tx_rollback')()

    def tx_select(self):
        if False:
            for i in range(10):
                print('nop')
        'Select standard transaction mode. This method sets the channel to use\n        standard transactions. The client must use this method at least once on\n        a channel before using the Commit or Rollback methods.\n\n        :returns: Deferred that fires on the Tx.SelectOk response\n        :rtype: Deferred\n        :raises ValueError:\n\n        '
        return self._wrap_channel_method('tx_select')()

class _TwistedConnectionAdapter(pika.connection.Connection):
    """A Twisted-specific implementation of a Pika Connection.

    NOTE: since `base_connection.BaseConnection`'s primary responsibility is
    management of the transport, we use `pika.connection.Connection` directly
    as our base class because this adapter uses a different transport
    management strategy.

    """

    def __init__(self, parameters, on_open_callback, on_open_error_callback, on_close_callback, custom_reactor):
        if False:
            while True:
                i = 10
        super().__init__(parameters=parameters, on_open_callback=on_open_callback, on_open_error_callback=on_open_error_callback, on_close_callback=on_close_callback, internal_connection_workflow=False)
        self._reactor = custom_reactor or reactor
        self._transport = None

    def _adapter_call_later(self, delay, callback):
        if False:
            while True:
                i = 10
        'Implement\n        :py:meth:`pika.connection.Connection._adapter_call_later()`.\n\n        '
        check_callback_arg(callback, 'callback')
        return _TimerHandle(self._reactor.callLater(delay, callback))

    def _adapter_remove_timeout(self, timeout_id):
        if False:
            return 10
        'Implement\n        :py:meth:`pika.connection.Connection._adapter_remove_timeout()`.\n\n        '
        timeout_id.cancel()

    def _adapter_add_callback_threadsafe(self, callback):
        if False:
            while True:
                i = 10
        'Implement\n        :py:meth:`pika.connection.Connection._adapter_add_callback_threadsafe()`.\n\n        '
        check_callback_arg(callback, 'callback')
        self._reactor.callFromThread(callback)

    def _adapter_connect_stream(self):
        if False:
            print('Hello World!')
        'Implement pure virtual\n        :py:ref:meth:`pika.connection.Connection._adapter_connect_stream()`\n         method.\n\n        NOTE: This should not be called due to our initialization of Connection\n        via `internal_connection_workflow=False`\n        '
        raise NotImplementedError

    def _adapter_disconnect_stream(self):
        if False:
            while True:
                i = 10
        'Implement pure virtual\n        :py:ref:meth:`pika.connection.Connection._adapter_disconnect_stream()`\n         method.\n\n        '
        self._transport.loseConnection()

    def _adapter_emit_data(self, data):
        if False:
            i = 10
            return i + 15
        'Implement pure virtual\n        :py:ref:meth:`pika.connection.Connection._adapter_emit_data()` method.\n\n        '
        self._transport.write(data)

    def connection_made(self, transport):
        if False:
            while True:
                i = 10
        'Introduces transport to protocol after transport is connected.\n\n        :param twisted.internet.interfaces.ITransport transport:\n        :raises Exception: Exception-based exception on error\n\n        '
        self._transport = transport
        self._on_stream_connected()

    def connection_lost(self, error):
        if False:
            i = 10
            return i + 15
        'Called upon loss or closing of TCP connection.\n\n        NOTE: `connection_made()` and `connection_lost()` are each called just\n        once and in that order. All other callbacks are called between them.\n\n        :param Failure: A Twisted Failure instance wrapping an exception.\n\n        '
        self._transport = None
        error = error.value
        if isinstance(error, twisted_error.ConnectionDone):
            self._error = error
            error = None
        LOGGER.log(logging.DEBUG if error is None else logging.ERROR, 'connection_lost: %r', error)
        self._on_stream_terminated(error)

    def data_received(self, data):
        if False:
            while True:
                i = 10
        'Called to deliver incoming data from the server to the protocol.\n\n        :param data: Non-empty data bytes.\n        :raises Exception: Exception-based exception on error\n\n        '
        self._on_data_available(data)

class TwistedProtocolConnection(protocol.Protocol):
    """A Pika-specific implementation of a Twisted Protocol. Allows using
    Twisted's non-blocking connectTCP/connectSSL methods for connecting to the
    server.

    TwistedProtocolConnection objects have a `ready` instance variable that's a
    Deferred which fires when the connection is ready to be used (the initial
    AMQP handshaking has been done). You *have* to wait for this Deferred to
    fire before requesting a channel.

    Once the connection is ready, you will be able to use the `closed` instance
    variable: a Deferred which fires when the connection is closed.

    Since it's Twisted handling connection establishing it does not accept
    connect callbacks, you have to implement that within Twisted. Also remember
    that the host, port and ssl values of the connection parameters are ignored
    because, yet again, it's Twisted who manages the connection.

    """

    def __init__(self, parameters=None, custom_reactor=None):
        if False:
            while True:
                i = 10
        self.ready = defer.Deferred()
        self.ready.addCallback(lambda _: self.connectionReady())
        self.closed = None
        self._impl = _TwistedConnectionAdapter(parameters=parameters, on_open_callback=self._on_connection_ready, on_open_error_callback=self._on_connection_failed, on_close_callback=self._on_connection_closed, custom_reactor=custom_reactor)
        self._calls = set()

    def channel(self, channel_number=None):
        if False:
            return 10
        'Create a new channel with the next available channel number or pass\n        in a channel number to use. Must be non-zero if you would like to\n        specify but it is recommended that you let Pika manage the channel\n        numbers.\n\n        :param int channel_number: The channel number to use, defaults to the\n                                   next available.\n        :returns: a Deferred that fires with an instance of a wrapper around\n            the Pika Channel class.\n        :rtype: Deferred\n\n        '
        d = defer.Deferred()
        self._impl.channel(channel_number, d.callback)
        self._calls.add(d)
        d.addCallback(self._clear_call, d)
        return d.addCallback(TwistedChannel)

    @property
    def is_open(self):
        if False:
            print('Hello World!')
        return self._impl.is_open

    @property
    def is_closed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._impl.is_closed

    def close(self, reply_code=200, reply_text='Normal shutdown'):
        if False:
            print('Hello World!')
        if not self._impl.is_closed:
            self._impl.close(reply_code, reply_text)
        return self.closed

    def dataReceived(self, data):
        if False:
            i = 10
            return i + 15
        self._impl.data_received(data)

    def connectionLost(self, reason=protocol.connectionDone):
        if False:
            return 10
        self._impl.connection_lost(reason)
        (d, self.ready) = (self.ready, None)
        if d:
            d.errback(reason)

    def makeConnection(self, transport):
        if False:
            for i in range(10):
                print('nop')
        self._impl.connection_made(transport)
        protocol.Protocol.makeConnection(self, transport)

    def connectionReady(self):
        if False:
            i = 10
            return i + 15
        'This method will be called when the underlying connection is ready.\n        '
        return self

    def _on_connection_ready(self, _connection):
        if False:
            for i in range(10):
                print('nop')
        (d, self.ready) = (self.ready, None)
        if d:
            self.closed = defer.Deferred()
            d.callback(None)

    def _on_connection_failed(self, _connection, _error_message=None):
        if False:
            return 10
        (d, self.ready) = (self.ready, None)
        if d:
            attempts = self._impl.params.connection_attempts
            exc = exceptions.AMQPConnectionError(attempts)
            d.errback(exc)

    def _on_connection_closed(self, _connection, exception):
        if False:
            print('Hello World!')
        for d in self._calls:
            d.errback(exception)
        self._calls = set()
        (d, self.closed) = (self.closed, None)
        if d:
            if isinstance(exception, Failure):
                exception = exception.value
            d.callback(exception)

    def _clear_call(self, ret, d):
        if False:
            while True:
                i = 10
        self._calls.discard(d)
        return ret

class _TimerHandle(nbio_interface.AbstractTimerReference):
    """This module's adaptation of `nbio_interface.AbstractTimerReference`.

    """

    def __init__(self, handle):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param twisted.internet.base.DelayedCall handle:\n        '
        self._handle = handle

    def cancel(self):
        if False:
            while True:
                i = 10
        if self._handle is not None:
            try:
                self._handle.cancel()
            except (twisted_error.AlreadyCalled, twisted_error.AlreadyCancelled):
                pass
            self._handle = None