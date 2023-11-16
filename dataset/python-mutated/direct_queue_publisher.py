from __future__ import absolute_import
import argparse
try:
    import pika
except ImportError:
    raise ImportError('Pika is not installed with StackStorm. Install it manually to use this tool.')

def main(queue, payload):
    if False:
        i = 10
        return i + 15
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', credentials=pika.credentials.PlainCredentials(username='guest', password='guest')))
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    channel.basic_publish(exchange='', routing_key=queue, body=payload)
    print('Sent %s' % payload)
    connection.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct queue publisher')
    parser.add_argument('--queue', required=True, help='Routing key to use')
    parser.add_argument('--payload', required=True, help='Message payload')
    args = parser.parse_args()
    main(queue=args.queue, payload=args.payload)