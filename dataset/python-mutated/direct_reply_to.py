"""
This example demonstrates RabbitMQ's "Direct reply-to" usage via
`pika.BlockingConnection`. See https://www.rabbitmq.com/direct-reply-to.html
for more info about this feature.
"""
import pika
SERVER_QUEUE = 'rpc.server.queue'

def main():
    if False:
        for i in range(10):
            print('nop')
    ' Here, Client sends "Marco" to RPC Server, and RPC Server replies with\n    "Polo".\n\n    NOTE Normally, the server would be running separately from the client, but\n    in this very simple example both are running in the same thread and sharing\n    connection and channel.\n\n    '
    with pika.BlockingConnection() as conn:
        channel = conn.channel()
        channel.queue_declare(queue=SERVER_QUEUE, exclusive=True, auto_delete=True)
        channel.basic_consume(SERVER_QUEUE, on_server_rx_rpc_request)
        channel.basic_consume('amq.rabbitmq.reply-to', on_client_rx_reply_from_server, auto_ack=True)
        channel.basic_publish(exchange='', routing_key=SERVER_QUEUE, body='Marco', properties=pika.BasicProperties(reply_to='amq.rabbitmq.reply-to'))
        channel.start_consuming()

def on_server_rx_rpc_request(ch, method_frame, properties, body):
    if False:
        for i in range(10):
            print('nop')
    print('RPC Server got request: %s' % body)
    ch.basic_publish('', routing_key=properties.reply_to, body='Polo')
    ch.basic_ack(delivery_tag=method_frame.delivery_tag)
    print('RPC Server says good bye')

def on_client_rx_reply_from_server(ch, _method_frame, _properties, body):
    if False:
        for i in range(10):
            print('nop')
    print('RPC Client got reply: %s' % body)
    print('RPC Client says bye')
    ch.close()
if __name__ == '__main__':
    main()