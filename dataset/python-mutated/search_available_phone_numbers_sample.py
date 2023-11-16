"""
FILE: search_available_phone_numbers_sample.py
DESCRIPTION:
    This sample demonstrates how to search for available numbers you can buy with the respective API.
USAGE:
    python search_available_phone_numbers_sample.py
    Set the environment variables with your own values before running the sample:
    1) COMMUNICATION_SAMPLES_CONNECTION_STRING - The connection string including your endpoint and 
        access key of your Azure Communication Service
"""
import os
from azure.communication.phonenumbers import PhoneNumbersClient, PhoneNumberType, PhoneNumberAssignmentType, PhoneNumberCapabilities, PhoneNumberCapabilityType
connection_str = os.getenv('COMMUNICATION_SAMPLES_CONNECTION_STRING')
phone_numbers_client = PhoneNumbersClient.from_connection_string(connection_str)

def search_available_phone_numbers():
    if False:
        print('Hello World!')
    capabilities = PhoneNumberCapabilities(calling=PhoneNumberCapabilityType.INBOUND, sms=PhoneNumberCapabilityType.INBOUND_OUTBOUND)
    poller = phone_numbers_client.begin_search_available_phone_numbers('US', PhoneNumberType.TOLL_FREE, PhoneNumberAssignmentType.APPLICATION, capabilities, polling=True)
    search_result = poller.result()
    print('Search id: ' + search_result.search_id)
    phone_number_list = search_result.phone_numbers
    print('Reserved phone numbers:')
    for phone_number in phone_number_list:
        print(phone_number)
if __name__ == '__main__':
    search_available_phone_numbers()