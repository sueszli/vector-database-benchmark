"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Pinpoint to
send SMS messages using a message template.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

def send_templated_sms_message(pinpoint_client, project_id, destination_number, message_type, origination_number, template_name, template_version):
    if False:
        return 10
    '\n    Sends an SMS message to a specific phone number using a pre-defined template.\n\n    :param pinpoint_client: A Boto3 Pinpoint client.\n    :param project_id: An Amazon Pinpoint project (application) ID.\n    :param destination_number: The phone number to send the message to.\n    :param message_type: The type of SMS message (promotional or transactional).\n    :param origination_number: The phone number that the message is sent from.\n    :param template_name: The name of the SMS template to use when sending the message.\n    :param template_version: The version number of the message template.\n\n    :return The ID of the message.\n    '
    try:
        response = pinpoint_client.send_messages(ApplicationId=project_id, MessageRequest={'Addresses': {destination_number: {'ChannelType': 'SMS'}}, 'MessageConfiguration': {'SMSMessage': {'MessageType': message_type, 'OriginationNumber': origination_number}}, 'TemplateConfiguration': {'SMSTemplate': {'Name': template_name, 'Version': template_version}}})
    except ClientError:
        logger.exception("Couldn't send message.")
        raise
    else:
        return response['MessageResponse']['Result'][destination_number]['MessageId']

def main():
    if False:
        i = 10
        return i + 15
    region = 'us-east-1'
    origination_number = '+18555550001'
    destination_number = '+14255550142'
    project_id = '7353f53e6885409fa32d07cedexample'
    message_type = 'TRANSACTIONAL'
    template_name = 'My_SMS_Template'
    template_version = '1'
    message_id = send_templated_sms_message(boto3.client('pinpoint', region_name=region), project_id, destination_number, message_type, origination_number, template_name, template_version)
    print(f'Message sent! Message ID: {message_id}.')
if __name__ == '__main__':
    main()