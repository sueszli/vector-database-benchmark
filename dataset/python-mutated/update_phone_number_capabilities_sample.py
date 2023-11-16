"""
FILE: update_phone_number_capabilities_sample.py
DESCRIPTION:
    This sample demonstrates how to updtae the capabilities of a phone number using your connection string.
USAGE:
    python update_phone_number_capabilities_sample.py
    Set the environment variables with your own values before running the sample:
    1) COMMUNICATION_SAMPLES_CONNECTION_STRING - The connection string including your endpoint and 
        access key of your Azure Communication Service
    2) AZURE_PHONE_NUMBER - The phone number you want to update
"""
import os
from azure.communication.phonenumbers import PhoneNumbersClient, PhoneNumberCapabilityType
connection_str = os.getenv('COMMUNICATION_SAMPLES_CONNECTION_STRING')
phone_number_to_update = os.getenv('AZURE_PHONE_NUMBER')
phone_numbers_client = PhoneNumbersClient.from_connection_string(connection_str)

def update_phone_number_capabilities():
    if False:
        return 10
    poller = phone_numbers_client.begin_update_phone_number_capabilities(phone_number_to_update, PhoneNumberCapabilityType.INBOUND_OUTBOUND, PhoneNumberCapabilityType.INBOUND, polling=True)
    poller.result()
    print('Status of the operation: ' + poller.status())
if __name__ == '__main__':
    update_phone_number_capabilities()