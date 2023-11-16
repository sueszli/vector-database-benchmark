"""
Purpose

Demonstrates subscribing Amazon Simple Queue Service (Amazon SQS)
queues to a FIFO (First-In-First-Out) Amazon Simple Notification Service (Amazon SNS) topic.
"""
import logging
import uuid
import boto3
import json
from botocore.exceptions import ClientError
from sns_basics import SnsWrapper
logger = logging.getLogger(__name__)

class FifoTopicWrapper:
    """Encapsulates Amazon SNS FIFO topic and subscription functions."""

    def __init__(self, sns_resource):
        if False:
            while True:
                i = 10
        '\n        :param sns_resource: A Boto3 Amazon SNS resource.\n        '
        self.sns_resource = sns_resource

    def create_fifo_topic(self, topic_name):
        if False:
            while True:
                i = 10
        '\n        Create a FIFO topic.\n        Topic names must be made up of only uppercase and lowercase ASCII letters,\n        numbers, underscores, and hyphens, and must be between 1 and 256 characters long.\n        For a FIFO topic, the name must end with the .fifo suffix.\n\n        :param topic_name: The name for the topic.\n        :return: The new topic.\n        '
        try:
            topic = self.sns_resource.create_topic(Name=topic_name, Attributes={'FifoTopic': str(True), 'ContentBasedDeduplication': str(False)})
            logger.info('Created FIFO topic with name=%s.', topic_name)
            return topic
        except ClientError as error:
            logger.exception("Couldn't create topic with name=%s!", topic_name)
            raise error

    @staticmethod
    def add_access_policy(queue, topic_arn):
        if False:
            i = 10
            return i + 15
        '\n        Add the necessary access policy to a queue, so\n        it can receive messages from a topic.\n\n        :param queue: The queue resource.\n        :param topic_arn: The ARN of the topic.\n        :return: None.\n        '
        try:
            queue.set_attributes(Attributes={'Policy': json.dumps({'Version': '2012-10-17', 'Statement': [{'Sid': 'test-sid', 'Effect': 'Allow', 'Principal': {'AWS': '*'}, 'Action': 'SQS:SendMessage', 'Resource': queue.attributes['QueueArn'], 'Condition': {'ArnLike': {'aws:SourceArn': topic_arn}}}]})})
            logger.info('Added trust policy to the queue.')
        except ClientError as error:
            logger.exception("Couldn't add trust policy to the queue!")
            raise error

    @staticmethod
    def subscribe_queue_to_topic(topic, queue_arn):
        if False:
            i = 10
            return i + 15
        '\n        Subscribe a queue to a topic.\n\n        :param topic: The topic resource.\n        :param queue_arn: The ARN of the queue.\n        :return: The subscription resource.\n        '
        try:
            subscription = topic.subscribe(Protocol='sqs', Endpoint=queue_arn)
            logger.info('The queue is subscribed to the topic.')
            return subscription
        except ClientError as error:
            logger.exception("Couldn't subscribe queue to topic!")
            raise error

    @staticmethod
    def publish_price_update(topic, payload, group_id):
        if False:
            while True:
                i = 10
        '\n        Compose and publish a message that updates the wholesale price.\n\n        :param topic: The topic to publish to.\n        :param payload: The message to publish.\n        :param group_id: The group ID for the message.\n        :return: The ID of the message.\n        '
        try:
            att_dict = {'business': {'DataType': 'String', 'StringValue': 'wholesale'}}
            dedup_id = uuid.uuid4()
            response = topic.publish(Subject='Price Update', Message=payload, MessageAttributes=att_dict, MessageGroupId=group_id, MessageDeduplicationId=str(dedup_id))
            message_id = response['MessageId']
            logger.info('Published message to topic %s.', topic.arn)
        except ClientError as error:
            logger.exception("Couldn't publish message to topic %s.", topic.arn)
            raise error
        return message_id

    @staticmethod
    def delete_queue(queue):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes an SQS queue. When run against an AWS account, it can take up to\n        60 seconds before the queue is actually deleted.\n\n        :param queue: The queue to delete.\n        :return: None\n        '
        try:
            queue.delete()
            logger.info('Deleted queue with URL=%s.', queue.url)
        except ClientError as error:
            logger.exception("Couldn't delete queue with URL=%s!", queue.url)
            raise error

def usage_demo():
    if False:
        i = 10
        return i + 15
    'Shows how to subscribe queues to a FIFO topic.'
    print('-' * 88)
    print('Welcome to the `Subscribe queues to a FIFO topic` demo!')
    print('-' * 88)
    sns = boto3.resource('sns')
    sqs = boto3.resource('sqs')
    fifo_topic_wrapper = FifoTopicWrapper(sns)
    sns_wrapper = SnsWrapper(sns)
    prefix = 'sqs-subscribe-demo-'
    queues = set()
    subscriptions = set()
    wholesale_queue = sqs.create_queue(QueueName=prefix + 'wholesale.fifo', Attributes={'MaximumMessageSize': str(4096), 'ReceiveMessageWaitTimeSeconds': str(10), 'VisibilityTimeout': str(300), 'FifoQueue': str(True), 'ContentBasedDeduplication': str(True)})
    queues.add(wholesale_queue)
    print(f'Created FIFO queue with URL: {wholesale_queue.url}.')
    retail_queue = sqs.create_queue(QueueName=prefix + 'retail.fifo', Attributes={'MaximumMessageSize': str(4096), 'ReceiveMessageWaitTimeSeconds': str(10), 'VisibilityTimeout': str(300), 'FifoQueue': str(True), 'ContentBasedDeduplication': str(True)})
    queues.add(retail_queue)
    print(f'Created FIFO queue with URL: {retail_queue.url}.')
    analytics_queue = sqs.create_queue(QueueName=prefix + 'analytics', Attributes={})
    queues.add(analytics_queue)
    print(f'Created standard queue with URL: {analytics_queue.url}.')
    topic = fifo_topic_wrapper.create_fifo_topic('price-updates-topic.fifo')
    print(f"Created FIFO topic: {topic.attributes['TopicArn']}.")
    for q in queues:
        fifo_topic_wrapper.add_access_policy(q, topic.attributes['TopicArn'])
    print(f"Added access policies for topic: {topic.attributes['TopicArn']}.")
    for q in queues:
        sub = fifo_topic_wrapper.subscribe_queue_to_topic(topic, q.attributes['QueueArn'])
        subscriptions.add(sub)
    print(f"Subscribed queues to topic: {topic.attributes['TopicArn']}.")
    input('Press Enter to publish a message to the topic.')
    message_id = fifo_topic_wrapper.publish_price_update(topic, '{"product": 214, "price": 79.99}', 'Consumables')
    print(f'Published price update with message ID: {message_id}.')
    input('Press Enter to clean up resources.')
    for s in subscriptions:
        sns_wrapper.delete_subscription(s)
    sns_wrapper.delete_topic(topic)
    for q in queues:
        fifo_topic_wrapper.delete_queue(q)
    print(f'Deleted subscriptions, queues, and topic.')
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()