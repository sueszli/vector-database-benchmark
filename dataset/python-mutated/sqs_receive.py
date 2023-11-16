from nameko.extensions import Entrypoint
from functools import partial
import boto3

class SqsReceive(Entrypoint):

    def __init__(self, url, region='eu-west-1', **kwargs):
        if False:
            print('Hello World!')
        self.url = url
        self.region = region
        super(SqsReceive, self).__init__(**kwargs)

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.client = boto3.client('sqs', region_name=self.region)

    def start(self):
        if False:
            while True:
                i = 10
        self.container.spawn_managed_thread(self.run, identifier='SqsReceiver.run')

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            response = self.client.receive_message(QueueUrl=self.url, WaitTimeSeconds=5)
            messages = response.get('Messages', ())
            for message in messages:
                self.handle_message(message)

    def handle_message(self, message):
        if False:
            while True:
                i = 10
        handle_result = partial(self.handle_result, message)
        args = (message['Body'],)
        kwargs = {}
        self.container.spawn_worker(self, args, kwargs, handle_result=handle_result)

    def handle_result(self, message, worker_ctx, result, exc_info):
        if False:
            while True:
                i = 10
        self.client.delete_message(QueueUrl=self.url, ReceiptHandle=message['ReceiptHandle'])
        return (result, exc_info)
receive = SqsReceive.decorator