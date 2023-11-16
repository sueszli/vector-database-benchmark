"""
# based on:
#  - txamqp-helpers by Dan Siemon <dan@coverfire.com> (March 2010)
#    http://git.coverfire.com/?p=txamqp-twistd.git;a=tree
#  - Post by Brian Chandler
#    https://groups.google.com/forum/#!topic/pika-python/o_deVmGondk
#  - Pika Documentation
#    https://pika.readthedocs.io/en/latest/examples/twisted_example.html


Fire up this test application via `twistd -ny twisted_service.py`

The application will answer to requests to exchange "foobar" and any of the
routing_key values: "request1", "request2", or "request3"
with messages to the same exchange, but with routing_key "response"

When a routing_key of "task" is used on the exchange "foobar",
the application can asynchronously run a maximum of 2 tasks at once
as defined by PREFETCH_COUNT
"""
import logging
import sys
from twisted.internet import protocol
from twisted.application import internet
from twisted.application import service
from twisted.internet.defer import inlineCallbacks
from twisted.internet import ssl, defer, task
from twisted.python import log
from twisted.internet import reactor
import pika
from pika import spec, DeliveryMode
from pika.adapters import twisted_connection
from pika.exchange_type import ExchangeType
PREFETCH_COUNT = 2

class PikaService(service.MultiService):
    name = 'amqp'

    def __init__(self, parameter):
        if False:
            for i in range(10):
                print('nop')
        service.MultiService.__init__(self)
        self.parameters = parameter

    def startService(self):
        if False:
            for i in range(10):
                print('nop')
        self.connect()
        service.MultiService.startService(self)

    def getFactory(self):
        if False:
            for i in range(10):
                print('nop')
        return self.services[0].factory

    def connect(self):
        if False:
            return 10
        f = PikaFactory(self.parameters)
        if self.parameters.ssl_options:
            s = ssl.ClientContextFactory()
            serv = internet.SSLClient(host=self.parameters.host, port=self.parameters.port, factory=f, contextFactory=s)
        else:
            serv = internet.TCPClient(host=self.parameters.host, port=self.parameters.port, factory=f)
        serv.factory = f
        f.service = serv
        name = '%s%s:%d' % ('ssl:' if self.parameters.ssl_options else '', self.parameters.host, self.parameters.port)
        serv.__repr__ = lambda : '<AMQP Connection to %s>' % name
        serv.setName(name)
        serv.setServiceParent(self)

class PikaProtocol(twisted_connection.TwistedProtocolConnection):
    connected = False
    name = 'AMQP:Protocol'

    def __init__(self, factory, parameters):
        if False:
            return 10
        super().__init__(parameters)
        self.factory = factory

    @inlineCallbacks
    def connectionReady(self):
        if False:
            i = 10
            return i + 15
        self._channel = (yield self.channel())
        yield self._channel.basic_qos(prefetch_count=PREFETCH_COUNT)
        self.connected = True
        yield self._channel.confirm_delivery()
        for (exchange, routing_key, callback) in self.factory.read_list:
            yield self.setup_read(exchange, routing_key, callback)
        self.send()

    @inlineCallbacks
    def read(self, exchange, routing_key, callback):
        if False:
            while True:
                i = 10
        'Add an exchange to the list of exchanges to read from.'
        if self.connected:
            yield self.setup_read(exchange, routing_key, callback)

    @inlineCallbacks
    def setup_read(self, exchange, routing_key, callback):
        if False:
            print('Hello World!')
        'This function does the work to read from an exchange.'
        if exchange:
            yield self._channel.exchange_declare(exchange=exchange, exchange_type=ExchangeType.topic, durable=True, auto_delete=False)
        yield self._channel.queue_declare(queue=routing_key, durable=True)
        if exchange:
            yield self._channel.queue_bind(queue=routing_key, exchange=exchange)
            yield self._channel.queue_bind(queue=routing_key, exchange=exchange, routing_key=routing_key)
        (queue, _consumer_tag) = (yield self._channel.basic_consume(queue=routing_key, auto_ack=False))
        d = queue.get()
        d.addCallback(self._read_item, queue, callback)
        d.addErrback(self._read_item_err)

    def _read_item(self, item, queue, callback):
        if False:
            while True:
                i = 10
        'Callback function which is called when an item is read.'
        d = queue.get()
        d.addCallback(self._read_item, queue, callback)
        d.addErrback(self._read_item_err)
        (channel, deliver, _props, msg) = item
        log.msg('%s (%s): %s' % (deliver.exchange, deliver.routing_key, repr(msg)), system='Pika:<=')
        d = defer.maybeDeferred(callback, item)
        d.addCallbacks(lambda _: channel.basic_ack(deliver.delivery_tag), lambda _: channel.basic_nack(deliver.delivery_tag))

    @staticmethod
    def _read_item_err(error):
        if False:
            return 10
        print(error)

    def send(self):
        if False:
            while True:
                i = 10
        'If connected, send all waiting messages.'
        if self.connected:
            while self.factory.queued_messages:
                (exchange, r_key, message) = self.factory.queued_messages.pop(0)
                self.send_message(exchange, r_key, message)

    @inlineCallbacks
    def send_message(self, exchange, routing_key, msg):
        if False:
            for i in range(10):
                print('nop')
        'Send a single message.'
        log.msg('%s (%s): %s' % (exchange, routing_key, repr(msg)), system='Pika:=>')
        yield self._channel.exchange_declare(exchange=exchange, exchange_type=ExchangeType.topic, durable=True, auto_delete=False)
        prop = spec.BasicProperties(delivery_mode=DeliveryMode.Persistent)
        try:
            yield self._channel.basic_publish(exchange=exchange, routing_key=routing_key, body=msg, properties=prop)
        except Exception as error:
            log.msg('Error while sending message: %s' % error, system=self.name)

class PikaFactory(protocol.ReconnectingClientFactory):
    name = 'AMQP:Factory'

    def __init__(self, parameters):
        if False:
            while True:
                i = 10
        self.parameters = parameters
        self.client = None
        self.queued_messages = []
        self.read_list = []

    def startedConnecting(self, connector):
        if False:
            print('Hello World!')
        log.msg('Started to connect.', system=self.name)

    def buildProtocol(self, addr):
        if False:
            print('Hello World!')
        self.resetDelay()
        log.msg('Connected', system=self.name)
        self.client = PikaProtocol(self, self.parameters)
        return self.client

    def clientConnectionLost(self, connector, reason):
        if False:
            for i in range(10):
                print('nop')
        log.msg('Lost connection.  Reason: %s' % reason.value, system=self.name)
        protocol.ReconnectingClientFactory.clientConnectionLost(self, connector, reason)

    def clientConnectionFailed(self, connector, reason):
        if False:
            while True:
                i = 10
        log.msg('Connection failed. Reason: %s' % reason.value, system=self.name)
        protocol.ReconnectingClientFactory.clientConnectionFailed(self, connector, reason)

    def send_message(self, exchange=None, routing_key=None, message=None):
        if False:
            for i in range(10):
                print('nop')
        self.queued_messages.append((exchange, routing_key, message))
        if self.client is not None:
            self.client.send()

    def read_messages(self, exchange, routing_key, callback):
        if False:
            for i in range(10):
                print('nop')
        'Configure an exchange to be read from.'
        self.read_list.append((exchange, routing_key, callback))
        if self.client is not None:
            self.client.read(exchange, routing_key, callback)
application = service.Application('pikaapplication')
ps = PikaService(pika.ConnectionParameters(host='localhost', virtual_host='/', credentials=pika.PlainCredentials('guest', 'guest')))
ps.setServiceParent(application)

class TestService(service.Service):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.amqp = None

    def task(self, _msg):
        if False:
            i = 10
            return i + 15
        '\n        Method for a time consuming task.\n\n        This function must return a deferred. If it is successfull,\n        a `basic.ack` will be sent to AMQP. If the task was not completed a\n        `basic.nack` will be sent. In this example it will always return\n        successfully after a 2 second pause.\n        '
        return task.deferLater(reactor, 2, lambda : log.msg('task completed'))

    def respond(self, msg):
        if False:
            print('Hello World!')
        self.amqp.send_message('foobar', 'response', msg[3])

    def startService(self):
        if False:
            for i in range(10):
                print('nop')
        amqp_service = self.parent.getServiceNamed('amqp')
        self.amqp = amqp_service.getFactory()
        self.amqp.read_messages('foobar', 'request1', self.respond)
        self.amqp.read_messages('foobar', 'request2', self.respond)
        self.amqp.read_messages('foobar', 'request3', self.respond)
        self.amqp.read_messages('foobar', 'task', self.task)
ts = TestService()
ts.setServiceParent(application)
observer = log.PythonLoggingObserver()
observer.start()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)