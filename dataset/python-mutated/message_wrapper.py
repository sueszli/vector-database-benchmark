"""
Purpose

Demonstrate basic message operations in Amazon Simple Queue Service (Amazon SQS).
"""
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import queue_wrapper
logger = logging.getLogger(__name__)
sqs = boto3.resource('sqs')

def send_message(queue, message_body, message_attributes=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send a message to an Amazon SQS queue.\n\n    :param queue: The queue that receives the message.\n    :param message_body: The body text of the message.\n    :param message_attributes: Custom attributes of the message. These are key-value\n                               pairs that can be whatever you want.\n    :return: The response from SQS that contains the assigned message ID.\n    '
    if not message_attributes:
        message_attributes = {}
    try:
        response = queue.send_message(MessageBody=message_body, MessageAttributes=message_attributes)
    except ClientError as error:
        logger.exception('Send message failed: %s', message_body)
        raise error
    else:
        return response

def send_messages(queue, messages):
    if False:
        while True:
            i = 10
    '\n    Send a batch of messages in a single request to an SQS queue.\n    This request may return overall success even when some messages were not sent.\n    The caller must inspect the Successful and Failed lists in the response and\n    resend any failed messages.\n\n    :param queue: The queue to receive the messages.\n    :param messages: The messages to send to the queue. These are simplified to\n                     contain only the message body and attributes.\n    :return: The response from SQS that contains the list of successful and failed\n             messages.\n    '
    try:
        entries = [{'Id': str(ind), 'MessageBody': msg['body'], 'MessageAttributes': msg['attributes']} for (ind, msg) in enumerate(messages)]
        response = queue.send_messages(Entries=entries)
        if 'Successful' in response:
            for msg_meta in response['Successful']:
                logger.info('Message sent: %s: %s', msg_meta['MessageId'], messages[int(msg_meta['Id'])]['body'])
        if 'Failed' in response:
            for msg_meta in response['Failed']:
                logger.warning('Failed to send: %s: %s', msg_meta['MessageId'], messages[int(msg_meta['Id'])]['body'])
    except ClientError as error:
        logger.exception('Send messages failed to queue: %s', queue)
        raise error
    else:
        return response

def receive_messages(queue, max_number, wait_time):
    if False:
        print('Hello World!')
    '\n    Receive a batch of messages in a single request from an SQS queue.\n\n    :param queue: The queue from which to receive messages.\n    :param max_number: The maximum number of messages to receive. The actual number\n                       of messages received might be less.\n    :param wait_time: The maximum time to wait (in seconds) before returning. When\n                      this number is greater than zero, long polling is used. This\n                      can result in reduced costs and fewer false empty responses.\n    :return: The list of Message objects received. These each contain the body\n             of the message and metadata and custom attributes.\n    '
    try:
        messages = queue.receive_messages(MessageAttributeNames=['All'], MaxNumberOfMessages=max_number, WaitTimeSeconds=wait_time)
        for msg in messages:
            logger.info('Received message: %s: %s', msg.message_id, msg.body)
    except ClientError as error:
        logger.exception("Couldn't receive messages from queue: %s", queue)
        raise error
    else:
        return messages

def delete_message(message):
    if False:
        i = 10
        return i + 15
    "\n    Delete a message from a queue. Clients must delete messages after they\n    are received and processed to remove them from the queue.\n\n    :param message: The message to delete. The message's queue URL is contained in\n                    the message's metadata.\n    :return: None\n    "
    try:
        message.delete()
        logger.info('Deleted message: %s', message.message_id)
    except ClientError as error:
        logger.exception("Couldn't delete message: %s", message.message_id)
        raise error

def delete_messages(queue, messages):
    if False:
        i = 10
        return i + 15
    '\n    Delete a batch of messages from a queue in a single request.\n\n    :param queue: The queue from which to delete the messages.\n    :param messages: The list of messages to delete.\n    :return: The response from SQS that contains the list of successful and failed\n             message deletions.\n    '
    try:
        entries = [{'Id': str(ind), 'ReceiptHandle': msg.receipt_handle} for (ind, msg) in enumerate(messages)]
        response = queue.delete_messages(Entries=entries)
        if 'Successful' in response:
            for msg_meta in response['Successful']:
                logger.info('Deleted %s', messages[int(msg_meta['Id'])].receipt_handle)
        if 'Failed' in response:
            for msg_meta in response['Failed']:
                logger.warning('Could not delete %s', messages[int(msg_meta['Id'])].receipt_handle)
    except ClientError:
        logger.exception("Couldn't delete messages from queue %s", queue)
    else:
        return response

def usage_demo():
    if False:
        for i in range(10):
            print('nop')
    '\n    Shows how to:\n    * Read the lines from this Python file and send the lines in\n      batches of 10 as messages to a queue.\n    * Receive the messages in batches until the queue is empty.\n    * Reassemble the lines of the file and verify they match the original file.\n    '

    def pack_message(msg_path, msg_body, msg_line):
        if False:
            for i in range(10):
                print('nop')
        return {'body': msg_body, 'attributes': {'path': {'StringValue': msg_path, 'DataType': 'String'}, 'line': {'StringValue': str(msg_line), 'DataType': 'String'}}}

    def unpack_message(msg):
        if False:
            return 10
        return (msg.message_attributes['path']['StringValue'], msg.body, int(msg.message_attributes['line']['StringValue']))
    print('-' * 88)
    print('Welcome to the Amazon Simple Queue Service (Amazon SQS) demo!')
    print('-' * 88)
    queue = queue_wrapper.create_queue('sqs-usage-demo-message-wrapper')
    with open(__file__) as file:
        lines = file.readlines()
    line = 0
    batch_size = 10
    received_lines = [None] * len(lines)
    print(f'Sending file lines in batches of {batch_size} as messages.')
    while line < len(lines):
        messages = [pack_message(__file__, lines[index], index) for index in range(line, min(line + batch_size, len(lines)))]
        line = line + batch_size
        send_messages(queue, messages)
        print('.', end='')
        sys.stdout.flush()
    print(f'Done. Sent {len(lines) - 1} messages.')
    print(f'Receiving, handling, and deleting messages in batches of {batch_size}.')
    more_messages = True
    while more_messages:
        received_messages = receive_messages(queue, batch_size, 2)
        print('.', end='')
        sys.stdout.flush()
        for message in received_messages:
            (path, body, line) = unpack_message(message)
            received_lines[line] = body
        if received_messages:
            delete_messages(queue, received_messages)
        else:
            more_messages = False
    print('Done.')
    if all([lines[index] == received_lines[index] for index in range(len(lines))]):
        print(f'Successfully reassembled all file lines!')
    else:
        print(f'Uh oh, some lines were missed!')
    queue.delete()
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()