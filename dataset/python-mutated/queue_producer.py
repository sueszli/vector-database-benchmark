"""
A utility script which sends test messages to a queue.
"""
from __future__ import absolute_import
import argparse
import eventlet
from kombu import Exchange
from st2common import config
from st2common.transport.publishers import PoolPublisher

def main(exchange, routing_key, payload):
    if False:
        while True:
            i = 10
    exchange = Exchange(exchange, type='topic')
    publisher = PoolPublisher()
    publisher.publish(payload=payload, exchange=exchange, routing_key=routing_key)
    eventlet.sleep(0.5)
if __name__ == '__main__':
    config.parse_args(args={})
    parser = argparse.ArgumentParser(description='Queue producer')
    parser.add_argument('--exchange', required=True, help='Exchange to publish the message to')
    parser.add_argument('--routing-key', required=True, help='Routing key to use')
    parser.add_argument('--payload', required=True, help='Message payload')
    args = parser.parse_args()
    main(exchange=args.exchange, routing_key=args.routing_key, payload=args.payload)