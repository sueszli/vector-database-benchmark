"""
This example demonstrates explicit setting of heartbeat and blocked connection
timeouts.

Starting with RabbitMQ 3.5.5, the broker's default hearbeat timeout decreased
from 580 seconds to 60 seconds. As a result, applications that perform lengthy
processing in the same thread that also runs their Pika connection may
experience unexpected dropped connections due to heartbeat timeout. Here, we
specify an explicit lower bound for heartbeat timeout.

When RabbitMQ broker is running out of certain resources, such as memory and
disk space, it may block connections that are performing resource-consuming
operations, such as publishing messages. Once a connection is blocked, RabbitMQ
stops reading from that connection's socket, so no commands from the client will
get through to te broker on that connection until the broker unblocks it. A
blocked connection may last for an indefinite period of time, stalling the
connection and possibly resulting in a hang (e.g., in BlockingConnection) until
the connection is unblocked. Blocked Connection Timeout is intended to interrupt
(i.e., drop) a connection that has been blocked longer than the given timeout
value.
"""
import pika

def main():
    if False:
        for i in range(10):
            print('nop')
    params = pika.ConnectionParameters(heartbeat=600, blocked_connection_timeout=300)
    conn = pika.BlockingConnection(params)
    chan = conn.channel()
    chan.basic_publish('', 'my-alphabet-queue', 'abc')
    conn.close()
if __name__ == '__main__':
    main()