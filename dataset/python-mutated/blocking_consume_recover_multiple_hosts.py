import functools
import random
import pika
from pika.exchange_type import ExchangeType

def on_message(ch, method_frame, _header_frame, body, userdata=None):
    if False:
        i = 10
        return i + 15
    print('Userdata: {} Message body: {}'.format(userdata, body))
    ch.basic_ack(delivery_tag=method_frame.delivery_tag)
credentials = pika.PlainCredentials('guest', 'guest')
params1 = pika.ConnectionParameters('localhost', port=5672, credentials=credentials)
params2 = pika.ConnectionParameters('localhost', port=5673, credentials=credentials)
params3 = pika.ConnectionParameters('localhost', port=5674, credentials=credentials)
params_all = [params1, params2, params3]
while True:
    try:
        random.shuffle(params_all)
        connection = pika.BlockingConnection(params_all)
        channel = connection.channel()
        channel.exchange_declare(exchange='test_exchange', exchange_type=ExchangeType.direct, passive=False, durable=True, auto_delete=False)
        channel.queue_declare(queue='standard', auto_delete=True)
        channel.queue_bind(queue='standard', exchange='test_exchange', routing_key='standard_key')
        channel.basic_qos(prefetch_count=1)
        on_message_callback = functools.partial(on_message, userdata='on_message_userdata')
        channel.basic_consume('standard', on_message_callback)
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()
        connection.close()
        break
    except pika.exceptions.ConnectionClosedByBroker:
        break
    except pika.exceptions.AMQPChannelError:
        break
    except pika.exceptions.AMQPConnectionError:
        continue