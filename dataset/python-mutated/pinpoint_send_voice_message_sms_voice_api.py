"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Pinpoint SMS and Voice API
to send synthesized voice messages.
"""
import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

def send_voice_message(sms_voice_client, origination_number, caller_id, destination_number, language_code, voice_id, ssml_message):
    if False:
        while True:
            i = 10
    "\n    Sends a voice message using speech synthesis provided by Amazon Polly.\n\n    :param sms_voice_client: A Boto3 PinpointSMSVoice client.\n    :param origination_number: The phone number that the message is sent from.\n                               The phone number must be associated with your Amazon\n                               Pinpoint account and be in E.164 format.\n    :param caller_id: The phone number that you want to appear on the recipient's\n                      device. The phone number must be associated with your Amazon\n                      Pinpoint account and be in E.164 format.\n    :param destination_number: The recipient's phone number. Specify the phone\n                               number in E.164 format.\n    :param language_code: The language to use when sending the message.\n    :param voice_id: The Amazon Polly voice that you want to use to send the message.\n    :param ssml_message: The content of the message. This example uses SSML to control\n                         certain aspects of the message, such as the volume and the\n                         speech rate. The message must not contain line breaks.\n    :return: The ID of the message.\n    "
    try:
        response = sms_voice_client.send_voice_message(DestinationPhoneNumber=destination_number, OriginationPhoneNumber=origination_number, CallerId=caller_id, Content={'SSMLMessage': {'LanguageCode': language_code, 'VoiceId': voice_id, 'Text': ssml_message}})
    except ClientError:
        logger.exception("Couldn't send message from %s to %s.", origination_number, destination_number)
        raise
    else:
        return response['MessageId']

def main():
    if False:
        i = 10
        return i + 15
    origination_number = '+12065550110'
    caller_id = '+12065550199'
    destination_number = '+12065550142'
    language_code = 'en-US'
    voice_id = 'Matthew'
    ssml_message = "<speak>This is a test message sent from <emphasis>Amazon Pinpoint</emphasis> using the <break strength='weak'/>AWS SDK for Python (Boto3). <amazon:effect phonation='soft'>Thank you for listening.</amazon:effect></speak>"
    print(f'Sending voice message from {origination_number} to {destination_number}.')
    message_id = send_voice_message(boto3.client('pinpoint-sms-voice'), origination_number, caller_id, destination_number, language_code, voice_id, ssml_message)
    print(f'Message sent!\nMessage ID: {message_id}')
if __name__ == '__main__':
    main()