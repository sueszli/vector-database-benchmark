import os
import boto3

def handler(event, context):
    if False:
        return 10
    endpoint_url = None
    if os.environ.get('AWS_ENDPOINT_URL'):
        endpoint_url = os.environ['AWS_ENDPOINT_URL']
    sqs = boto3.client('sqs', endpoint_url=endpoint_url, region_name=event['region_name'], verify=False)
    queue_url = sqs.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    rs = sqs.send_message(QueueUrl=queue_url, MessageBody=event['message'])
    return rs['MessageId']