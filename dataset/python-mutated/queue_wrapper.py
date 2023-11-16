"""
Purpose

Demonstrate basic queue operations in Amazon Simple Queue Service (Amazon SQS).
Learn how to create, get, and remove standard, FIFO, and dead-letter queues.
Usage is shown in the test/test_queue_wrapper.py file.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)
sqs = boto3.resource('sqs')

def create_queue(name, attributes=None):
    if False:
        while True:
            i = 10
    "\n    Creates an Amazon SQS queue.\n\n    :param name: The name of the queue. This is part of the URL assigned to the queue.\n    :param attributes: The attributes of the queue, such as maximum message size or\n                       whether it's a FIFO queue.\n    :return: A Queue object that contains metadata about the queue and that can be used\n             to perform queue operations like sending and receiving messages.\n    "
    if not attributes:
        attributes = {}
    try:
        queue = sqs.create_queue(QueueName=name, Attributes=attributes)
        logger.info("Created queue '%s' with URL=%s", name, queue.url)
    except ClientError as error:
        logger.exception("Couldn't create queue named '%s'.", name)
        raise error
    else:
        return queue

def get_queue(name):
    if False:
        while True:
            i = 10
    '\n    Gets an SQS queue by name.\n\n    :param name: The name that was used to create the queue.\n    :return: A Queue object.\n    '
    try:
        queue = sqs.get_queue_by_name(QueueName=name)
        logger.info("Got queue '%s' with URL=%s", name, queue.url)
    except ClientError as error:
        logger.exception("Couldn't get queue named %s.", name)
        raise error
    else:
        return queue

def get_queues(prefix=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets a list of SQS queues. When a prefix is specified, only queues with names\n    that start with the prefix are returned.\n\n    :param prefix: The prefix used to restrict the list of returned queues.\n    :return: A list of Queue objects.\n    '
    if prefix:
        queue_iter = sqs.queues.filter(QueueNamePrefix=prefix)
    else:
        queue_iter = sqs.queues.all()
    queues = list(queue_iter)
    if queues:
        logger.info('Got queues: %s', ', '.join([q.url for q in queues]))
    else:
        logger.warning('No queues found.')
    return queues

def remove_queue(queue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Removes an SQS queue. When run against an AWS account, it can take up to\n    60 seconds before the queue is actually deleted.\n\n    :param queue: The queue to delete.\n    :return: None\n    '
    try:
        queue.delete()
        logger.info('Deleted queue with URL=%s.', queue.url)
    except ClientError as error:
        logger.exception("Couldn't delete queue with URL=%s!", queue.url)
        raise error

def usage_demo():
    if False:
        return 10
    'Shows how to create, list, and delete queues.'
    print('-' * 88)
    print('Welcome to the Amazon Simple Queue Service (Amazon SQS) demo!')
    print('-' * 88)
    prefix = 'sqs-usage-demo-'
    river_queue = create_queue(prefix + 'peculiar-river', {'MaximumMessageSize': str(1024), 'ReceiveMessageWaitTimeSeconds': str(20)})
    print(f'Created queue with URL: {river_queue.url}.')
    lake_queue = create_queue(prefix + 'strange-lake.fifo', {'MaximumMessageSize': str(4096), 'ReceiveMessageWaitTimeSeconds': str(10), 'VisibilityTimeout': str(300), 'FifoQueue': str(True), 'ContentBasedDeduplication': str(True)})
    print(f'Created queue with URL: {lake_queue.url}.')
    stream_queue = create_queue(prefix + 'boring-stream')
    print(f'Created queue with URL: {stream_queue.url}.')
    alias_queue = get_queue(prefix + 'peculiar-river')
    print(f'Got queue with URL: {alias_queue.url}.')
    remove_queue(stream_queue)
    print(f'Removed queue with URL: {stream_queue.url}.')
    queues = get_queues(prefix=prefix)
    print(f'Got {len(queues)} queues.')
    for queue in queues:
        remove_queue(queue)
        print(f'Removed queue with URL: {queue.url}.')
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()