"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Simple Notification
Service (Amazon SNS) to create notification topics, add subscribers, and publish
messages.
"""
import json
import logging
import time
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class SnsWrapper:
    """Encapsulates Amazon SNS topic and subscription functions."""

    def __init__(self, sns_resource):
        if False:
            print('Hello World!')
        '\n        :param sns_resource: A Boto3 Amazon SNS resource.\n        '
        self.sns_resource = sns_resource

    def create_topic(self, name):
        if False:
            return 10
        '\n        Creates a notification topic.\n\n        :param name: The name of the topic to create.\n        :return: The newly created topic.\n        '
        try:
            topic = self.sns_resource.create_topic(Name=name)
            logger.info('Created topic %s with ARN %s.', name, topic.arn)
        except ClientError:
            logger.exception("Couldn't create topic %s.", name)
            raise
        else:
            return topic

    def list_topics(self):
        if False:
            print('Hello World!')
        '\n        Lists topics for the current account.\n\n        :return: An iterator that yields the topics.\n        '
        try:
            topics_iter = self.sns_resource.topics.all()
            logger.info('Got topics.')
        except ClientError:
            logger.exception("Couldn't get topics.")
            raise
        else:
            return topics_iter

    @staticmethod
    def delete_topic(topic):
        if False:
            while True:
                i = 10
        '\n        Deletes a topic. All subscriptions to the topic are also deleted.\n        '
        try:
            topic.delete()
            logger.info('Deleted topic %s.', topic.arn)
        except ClientError:
            logger.exception("Couldn't delete topic %s.", topic.arn)
            raise

    @staticmethod
    def subscribe(topic, protocol, endpoint):
        if False:
            print('Hello World!')
        "\n        Subscribes an endpoint to the topic. Some endpoint types, such as email,\n        must be confirmed before their subscriptions are active. When a subscription\n        is not confirmed, its Amazon Resource Number (ARN) is set to\n        'PendingConfirmation'.\n\n        :param topic: The topic to subscribe to.\n        :param protocol: The protocol of the endpoint, such as 'sms' or 'email'.\n        :param endpoint: The endpoint that receives messages, such as a phone number\n                         (in E.164 format) for SMS messages, or an email address for\n                         email messages.\n        :return: The newly added subscription.\n        "
        try:
            subscription = topic.subscribe(Protocol=protocol, Endpoint=endpoint, ReturnSubscriptionArn=True)
            logger.info('Subscribed %s %s to topic %s.', protocol, endpoint, topic.arn)
        except ClientError:
            logger.exception("Couldn't subscribe %s %s to topic %s.", protocol, endpoint, topic.arn)
            raise
        else:
            return subscription

    def list_subscriptions(self, topic=None):
        if False:
            return 10
        '\n        Lists subscriptions for the current account, optionally limited to a\n        specific topic.\n\n        :param topic: When specified, only subscriptions to this topic are returned.\n        :return: An iterator that yields the subscriptions.\n        '
        try:
            if topic is None:
                subs_iter = self.sns_resource.subscriptions.all()
            else:
                subs_iter = topic.subscriptions.all()
            logger.info('Got subscriptions.')
        except ClientError:
            logger.exception("Couldn't get subscriptions.")
            raise
        else:
            return subs_iter

    @staticmethod
    def add_subscription_filter(subscription, attributes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a filter policy to a subscription. A filter policy is a key and a\n        list of values that are allowed. When a message is published, it must have an\n        attribute that passes the filter or it will not be sent to the subscription.\n\n        :param subscription: The subscription the filter policy is attached to.\n        :param attributes: A dictionary of key-value pairs that define the filter.\n        '
        try:
            att_policy = {key: [value] for (key, value) in attributes.items()}
            subscription.set_attributes(AttributeName='FilterPolicy', AttributeValue=json.dumps(att_policy))
            logger.info('Added filter to subscription %s.', subscription.arn)
        except ClientError:
            logger.exception("Couldn't add filter to subscription %s.", subscription.arn)
            raise

    @staticmethod
    def delete_subscription(subscription):
        if False:
            print('Hello World!')
        '\n        Unsubscribes and deletes a subscription.\n        '
        try:
            subscription.delete()
            logger.info('Deleted subscription %s.', subscription.arn)
        except ClientError:
            logger.exception("Couldn't delete subscription %s.", subscription.arn)
            raise

    def publish_text_message(self, phone_number, message):
        if False:
            i = 10
            return i + 15
        '\n        Publishes a text message directly to a phone number without need for a\n        subscription.\n\n        :param phone_number: The phone number that receives the message. This must be\n                             in E.164 format. For example, a United States phone\n                             number might be +12065550101.\n        :param message: The message to send.\n        :return: The ID of the message.\n        '
        try:
            response = self.sns_resource.meta.client.publish(PhoneNumber=phone_number, Message=message)
            message_id = response['MessageId']
            logger.info('Published message to %s.', phone_number)
        except ClientError:
            logger.exception("Couldn't publish message to %s.", phone_number)
            raise
        else:
            return message_id

    @staticmethod
    def publish_message(topic, message, attributes):
        if False:
            while True:
                i = 10
        '\n        Publishes a message, with attributes, to a topic. Subscriptions can be filtered\n        based on message attributes so that a subscription receives messages only\n        when specified attributes are present.\n\n        :param topic: The topic to publish to.\n        :param message: The message to publish.\n        :param attributes: The key-value attributes to attach to the message. Values\n                           must be either `str` or `bytes`.\n        :return: The ID of the message.\n        '
        try:
            att_dict = {}
            for (key, value) in attributes.items():
                if isinstance(value, str):
                    att_dict[key] = {'DataType': 'String', 'StringValue': value}
                elif isinstance(value, bytes):
                    att_dict[key] = {'DataType': 'Binary', 'BinaryValue': value}
            response = topic.publish(Message=message, MessageAttributes=att_dict)
            message_id = response['MessageId']
            logger.info('Published message with attributes %s to topic %s.', attributes, topic.arn)
        except ClientError:
            logger.exception("Couldn't publish message to topic %s.", topic.arn)
            raise
        else:
            return message_id

    @staticmethod
    def publish_multi_message(topic, subject, default_message, sms_message, email_message):
        if False:
            i = 10
            return i + 15
        '\n        Publishes a multi-format message to a topic. A multi-format message takes\n        different forms based on the protocol of the subscriber. For example,\n        an SMS subscriber might receive a short, text-only version of the message\n        while an email subscriber could receive an HTML version of the message.\n\n        :param topic: The topic to publish to.\n        :param subject: The subject of the message.\n        :param default_message: The default version of the message. This version is\n                                sent to subscribers that have protocols that are not\n                                otherwise specified in the structured message.\n        :param sms_message: The version of the message sent to SMS subscribers.\n        :param email_message: The version of the message sent to email subscribers.\n        :return: The ID of the message.\n        '
        try:
            message = {'default': default_message, 'sms': sms_message, 'email': email_message}
            response = topic.publish(Message=json.dumps(message), Subject=subject, MessageStructure='json')
            message_id = response['MessageId']
            logger.info('Published multi-format message to topic %s.', topic.arn)
        except ClientError:
            logger.exception("Couldn't publish message to topic %s.", topic.arn)
            raise
        else:
            return message_id

def usage_demo():
    if False:
        return 10
    print('-' * 88)
    print('Welcome to the Amazon Simple Notification Service (Amazon SNS) demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sns_wrapper = SnsWrapper(boto3.resource('sns'))
    topic_name = f'demo-basics-topic-{time.time_ns()}'
    print(f'Creating topic {topic_name}.')
    topic = sns_wrapper.create_topic(topic_name)
    phone_number = input('Enter a phone number (in E.164 format) that can receive SMS messages: ')
    if phone_number != '':
        print(f'Sending an SMS message directly from SNS to {phone_number}.')
        sns_wrapper.publish_text_message(phone_number, 'Hello from the SNS demo!')
    email = input(f'Enter an email address to subscribe to {topic_name} and receive a message: ')
    if email != '':
        print(f'Subscribing {email} to {topic_name}.')
        email_sub = sns_wrapper.subscribe(topic, 'email', email)
        answer = input(f'Confirmation email sent to {email}. To receive SNS messages, follow the instructions in the email. When confirmed, press Enter to continue.')
        while email_sub.attributes['PendingConfirmation'] == 'true' and answer.lower() != 's':
            answer = input(f"Email address {email} is not confirmed. Follow the instructions in the email to confirm and receive SNS messages. Press Enter when confirmed or enter 's' to skip. ")
            email_sub.reload()
    phone_sub = None
    if phone_number != '':
        print(f'Subscribing {phone_number} to {topic_name}. Phone numbers do not require confirmation.')
        phone_sub = sns_wrapper.subscribe(topic, 'sms', phone_number)
    if phone_number != '' or email != '':
        print(f'Publishing a multi-format message to {topic_name}. Multi-format messages contain different messages for different kinds of endpoints.')
        sns_wrapper.publish_multi_message(topic, 'SNS multi-format demo', 'This is the default message.', 'This is the SMS version of the message.', 'This is the email version of the message.')
    if phone_sub is not None:
        mobile_key = 'mobile'
        friendly = 'friendly'
        print(f"Adding a filter policy to the {phone_number} subscription to send only messages with a '{mobile_key}' attribute of '{friendly}'.")
        sns_wrapper.add_subscription_filter(phone_sub, {mobile_key: friendly})
        print(f'Publishing a message with a {mobile_key}: {friendly} attribute.')
        sns_wrapper.publish_message(topic, 'Hello! This message is mobile friendly.', {mobile_key: friendly})
        not_friendly = 'not-friendly'
        print(f'Publishing a message with a {mobile_key}: {not_friendly} attribute.')
        sns_wrapper.publish_message(topic, "Hey. This message is not mobile friendly, so you shouldn't get it on your phone.", {mobile_key: not_friendly})
    print(f'Getting subscriptions to {topic_name}.')
    topic_subs = sns_wrapper.list_subscriptions(topic)
    for sub in topic_subs:
        print(f'{sub.arn}')
    print(f'Deleting subscriptions and {topic_name}.')
    for sub in topic_subs:
        if sub.arn != 'PendingConfirmation':
            sns_wrapper.delete_subscription(sub)
    sns_wrapper.delete_topic(topic)
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()