"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Pinpoint to
send SMS messages.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

def send_sms_message(pinpoint_client, app_id, origination_number, destination_number, message, message_type):
    if False:
        return 10
    "\n    Sends an SMS message with Amazon Pinpoint.\n\n    :param pinpoint_client: A Boto3 Pinpoint client.\n    :param app_id: The Amazon Pinpoint project/application ID to use when you send\n                   this message. The SMS channel must be enabled for the project or\n                   application.\n    :param destination_number: The recipient's phone number in E.164 format.\n    :param origination_number: The phone number to send the message from. This phone\n                               number must be associated with your Amazon Pinpoint\n                               account and be in E.164 format.\n    :param message: The content of the SMS message.\n    :param message_type: The type of SMS message that you want to send. If you send\n                         time-sensitive content, specify TRANSACTIONAL. If you send\n                         marketing-related content, specify PROMOTIONAL.\n    :return: The ID of the message.\n    "
    try:
        response = pinpoint_client.send_messages(ApplicationId=app_id, MessageRequest={'Addresses': {destination_number: {'ChannelType': 'SMS'}}, 'MessageConfiguration': {'SMSMessage': {'Body': message, 'MessageType': message_type, 'OriginationNumber': origination_number}}})
    except ClientError:
        logger.exception("Couldn't send message.")
        raise
    else:
        return response['MessageResponse']['Result'][destination_number]['MessageId']

def main():
    if False:
        i = 10
        return i + 15
    app_id = 'ce796be37f32f178af652b26eexample'
    origination_number = '+12065550199'
    destination_number = '+14255550142'
    message = 'This is a sample message sent from Amazon Pinpoint by using the AWS SDK for Python (Boto 3).'
    message_type = 'TRANSACTIONAL'
    print('Sending SMS message.')
    message_id = send_sms_message(boto3.client('pinpoint'), app_id, origination_number, destination_number, message, message_type)
    print(f'Message sent! Message ID: {message_id}.')
if __name__ == '__main__':
    main()