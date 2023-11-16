"""
A utility script which listens on queue for messages and prints them to stdout.
"""
from __future__ import absolute_import
import random
import argparse
from pprint import pprint
from kombu.mixins import ConsumerMixin
from kombu import Exchange, Queue
from st2common import config
from st2common.transport import utils as transport_utils

class QueueConsumer(ConsumerMixin):

    def __init__(self, connection, queue):
        if False:
            return 10
        self.connection = connection
        self.queue = queue

    def get_consumers(self, Consumer, channel):
        if False:
            print('Hello World!')
        return [Consumer(queues=[self.queue], accept=['pickle'], callbacks=[self.process_task])]

    def process_task(self, body, message):
        if False:
            for i in range(10):
                print('nop')
        print('===================================================')
        print('Received message')
        print('message.properties:')
        pprint(message.properties)
        print('message.delivery_info:')
        pprint(message.delivery_info)
        print('body:')
        pprint(body)
        print('===================================================')
        message.ack()

def main(queue, exchange, routing_key='#'):
    if False:
        i = 10
        return i + 15
    exchange = Exchange(exchange, type='topic')
    queue = Queue(name=queue, exchange=exchange, routing_key=routing_key, auto_delete=True)
    with transport_utils.get_connection() as connection:
        connection.connect()
        watcher = QueueConsumer(connection=connection, queue=queue)
        watcher.run()
if __name__ == '__main__':
    config.parse_args(args={})
    parser = argparse.ArgumentParser(description='Queue consumer')
    parser.add_argument('--exchange', required=True, help='Exchange to listen on')
    parser.add_argument('--routing-key', default='#', help='Routing key')
    args = parser.parse_args()
    queue_name = args.exchange + str(random.randint(1, 10000))
    main(queue=queue_name, exchange=args.exchange, routing_key=args.routing_key)