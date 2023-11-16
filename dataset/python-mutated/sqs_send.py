from nameko.extensions import DependencyProvider
import boto3

class SqsSend(DependencyProvider):

    def __init__(self, url, region='eu-west-1', **kwargs):
        if False:
            while True:
                i = 10
        self.url = url
        self.region = region
        super(SqsSend, self).__init__(**kwargs)

    def setup(self):
        if False:
            while True:
                i = 10
        self.client = boto3.client('sqs', region_name=self.region)

    def get_dependency(self, worker_ctx):
        if False:
            while True:
                i = 10

        def send_message(payload):
            if False:
                return 10
            self.client.send_message(QueueUrl=self.url, MessageBody=payload)
        return send_message