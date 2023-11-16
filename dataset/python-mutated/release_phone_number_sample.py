"""
FILE: release_phone_number_sample.py
DESCRIPTION:
    This sample demonstrates how to release a previously acquired phone number using your connection string.
USAGE:
    python release_phone_number_sample.py
    Set the environment variables with your own values before running the sample:
    1) COMMUNICATION_SAMPLES_CONNECTION_STRING - The connection string including your endpoint and 
        access key of your Azure Communication Service
    2) AZURE_PHONE_NUMBER_TO_RELEASE - The phone number you want to release
"""
import os
from azure.communication.phonenumbers import PhoneNumbersClient
connection_str = os.getenv('COMMUNICATION_SAMPLES_CONNECTION_STRING')
phone_number_to_release = os.getenv('AZURE_PHONE_NUMBER_TO_RELEASE')
phone_numbers_client = PhoneNumbersClient.from_connection_string(connection_str)

def release_phone_number():
    if False:
        for i in range(10):
            print('nop')
    poller = phone_numbers_client.begin_release_phone_number(phone_number_to_release)
    poller.result()
    print('Status of the operation: ' + poller.status())
if __name__ == '__main__':
    release_phone_number()