import threading
from time import sleep
from pika import ConnectionParameters, BlockingConnection, PlainCredentials

class Publisher(threading.Thread):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.daemon = True
        self.is_running = True
        self.name = 'Publisher'
        self.queue = 'downstream_queue'
        credentials = PlainCredentials('guest', 'guest')
        parameters = ConnectionParameters('localhost', credentials=credentials)
        self.connection = BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue, auto_delete=True)

    def run(self):
        if False:
            print('Hello World!')
        while self.is_running:
            self.connection.process_data_events(time_limit=1)

    def _publish(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.channel.basic_publish('', self.queue, body=message.encode())

    def publish(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.connection.add_callback_threadsafe(lambda : self._publish(message))

    def stop(self):
        if False:
            i = 10
            return i + 15
        print('Stopping...')
        self.is_running = False
        self.connection.process_data_events(time_limit=1)
        if self.connection.is_open:
            self.connection.close()
        print('Stopped')
if __name__ == '__main__':
    publisher = Publisher()
    publisher.start()
    try:
        for i in range(9999):
            msg = f'Message {i}'
            print(f'Publishing: {msg!r}')
            publisher.publish(msg)
            sleep(1)
    except KeyboardInterrupt:
        publisher.stop()
    finally:
        publisher.join()