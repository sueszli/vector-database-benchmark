import functools
import logging
import threading
import time
import pika
from pika.exchange_type import ExchangeType
LOG_FORMAT = '%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s'
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

def ack_message(ch, delivery_tag):
    if False:
        for i in range(10):
            print('nop')
    'Note that `ch` must be the same pika channel instance via which\n    the message being ACKed was retrieved (AMQP protocol constraint).\n    '
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        pass

def do_work(ch, delivery_tag, body):
    if False:
        i = 10
        return i + 15
    thread_id = threading.get_ident()
    LOGGER.info('Thread id: %s Delivery tag: %s Message body: %s', thread_id, delivery_tag, body)
    time.sleep(10)
    cb = functools.partial(ack_message, ch, delivery_tag)
    ch.connection.add_callback_threadsafe(cb)

def on_message(ch, method_frame, _header_frame, body, args):
    if False:
        for i in range(10):
            print('nop')
    thrds = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(ch, delivery_tag, body))
    t.start()
    thrds.append(t)
credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('localhost', credentials=credentials, heartbeat=5)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.exchange_declare(exchange='test_exchange', exchange_type=ExchangeType.direct, passive=False, durable=True, auto_delete=False)
channel.queue_declare(queue='standard', auto_delete=True)
channel.queue_bind(queue='standard', exchange='test_exchange', routing_key='standard_key')
channel.basic_qos(prefetch_count=1)
threads = []
on_message_callback = functools.partial(on_message, args=threads)
channel.basic_consume(on_message_callback=on_message_callback, queue='standard')
try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()
for thread in threads:
    thread.join()
connection.close()