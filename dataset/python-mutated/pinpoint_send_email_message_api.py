"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Pinpoint to
send email.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

def send_email_message(pinpoint_client, app_id, sender, to_addresses, char_set, subject, html_message, text_message):
    if False:
        while True:
            i = 10
    '\n    Sends an email message with HTML and plain text versions.\n\n    :param pinpoint_client: A Boto3 Pinpoint client.\n    :param app_id: The Amazon Pinpoint project ID to use when you send this message.\n    :param sender: The "From" address. This address must be verified in\n                   Amazon Pinpoint in the AWS Region you\'re using to send email.\n    :param to_addresses: The addresses on the "To" line. If your Amazon Pinpoint account\n                         is in the sandbox, these addresses must be verified.\n    :param char_set: The character encoding to use for the subject line and message\n                     body of the email.\n    :param subject: The subject line of the email.\n    :param html_message: The body of the email for recipients whose email clients can\n                         display HTML content.\n    :param text_message: The body of the email for recipients whose email clients\n                         don\'t support HTML content.\n    :return: A dict of to_addresses and their message IDs.\n    '
    try:
        response = pinpoint_client.send_messages(ApplicationId=app_id, MessageRequest={'Addresses': {to_address: {'ChannelType': 'EMAIL'} for to_address in to_addresses}, 'MessageConfiguration': {'EmailMessage': {'FromAddress': sender, 'SimpleEmail': {'Subject': {'Charset': char_set, 'Data': subject}, 'HtmlPart': {'Charset': char_set, 'Data': html_message}, 'TextPart': {'Charset': char_set, 'Data': text_message}}}}})
    except ClientError:
        logger.exception("Couldn't send email.")
        raise
    else:
        return {to_address: message['MessageId'] for (to_address, message) in response['MessageResponse']['Result'].items()}

def main():
    if False:
        for i in range(10):
            print('nop')
    app_id = 'ce796be37f32f178af652b26eexample'
    sender = 'sender@example.com'
    to_address = 'recipient@example.com'
    char_set = 'UTF-8'
    subject = 'Amazon Pinpoint Test (SDK for Python (Boto3))'
    text_message = 'Amazon Pinpoint Test (SDK for Python)\n    -------------------------------------\n    This email was sent with Amazon Pinpoint using the AWS SDK for Python (Boto3).\n    For more information, see https://aws.amazon.com/sdk-for-python/\n                '
    html_message = "<html>\n    <head></head>\n    <body>\n      <h1>Amazon Pinpoint Test (SDK for Python (Boto3)</h1>\n      <p>This email was sent with\n        <a href='https://aws.amazon.com/pinpoint/'>Amazon Pinpoint</a> using the\n        <a href='https://aws.amazon.com/sdk-for-python/'>\n          AWS SDK for Python (Boto3)</a>.</p>\n    </body>\n    </html>\n                "
    print('Sending email.')
    message_ids = send_email_message(boto3.client('pinpoint'), app_id, sender, [to_address], char_set, subject, html_message, text_message)
    print(f'Message sent! Message IDs: {message_ids}')
if __name__ == '__main__':
    main()