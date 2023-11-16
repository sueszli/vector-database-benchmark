"""
FILE: purchase_phone_number_sample.py
DESCRIPTION:
    This sample demonstrates how to purchase a phone number using the search id you got from the search_available_phone_number API
USAGE:
    python purchase_phone_number_sample.py
    Set the environment variables with your own values before running the sample:
    1) COMMUNICATION_SAMPLES_CONNECTION_STRING - The connection string including your endpoint and 
        access key of your Azure Communication Service    
    2) AZURE_COMMUNICATION_SERVICE_SEARCH_ID_TO_PURCHASE - The search id for the phone number you 
        reserved and want to purchase
"""
import os
from azure.communication.phonenumbers import PhoneNumbersClient
connection_str = os.getenv('COMMUNICATION_SAMPLES_CONNECTION_STRING')
search_id = os.getenv('AZURE_COMMUNICATION_SERVICE_SEARCH_ID_TO_PURCHASE')
phone_numbers_client = PhoneNumbersClient.from_connection_string(connection_str)

def purchase_phone_number():
    if False:
        for i in range(10):
            print('nop')
    poller = phone_numbers_client.begin_purchase_phone_numbers(search_id, polling=True)
    poller.result()
    print('Result from the purchase operation: ' + poller.status())
if __name__ == '__main__':
    purchase_phone_number()