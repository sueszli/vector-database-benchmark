"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Key Management Service (AWS KMS)
to create, list, and manage keys.
"""
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class KeyManager:

    def __init__(self, kms_client):
        if False:
            return 10
        self.kms_client = kms_client
        self.created_keys = []

    def create_key(self):
        if False:
            return 10
        '\n        Creates a key (or multiple keys) with a user-provided description.\n        '
        answer = 'y'
        while answer.lower() == 'y':
            key_desc = input("\nLet's create a key. Describe it for me: ")
            if not key_desc:
                key_desc = 'Key management demo key'
            try:
                key = self.kms_client.create_key(Description=key_desc)['KeyMetadata']
            except ClientError as err:
                logging.error("Couldn't create your key. Here's why: %s", err.response['Error']['Message'])
                raise
            else:
                print('Key created:')
                pprint(key)
                self.created_keys.append(key)
                answer = input('Create another (y/n)? ')

    def list_keys(self):
        if False:
            print('Hello World!')
        '\n        Lists the keys for the current account by using a paginator.\n        '
        try:
            page_size = 10
            print("\nLet's list your keys.")
            key_paginator = self.kms_client.get_paginator('list_keys')
            for key_page in key_paginator.paginate(PaginationConfig={'PageSize': 10}):
                print(f"Here are {len(key_page['Keys'])} keys:")
                pprint(key_page['Keys'])
                if key_page['Truncated']:
                    answer = input(f'Do you want to see the next {page_size} keys (y/n)? ')
                    if answer.lower() != 'y':
                        break
                else:
                    print("That's all your keys!")
        except ClientError as err:
            logging.error("Couldn't list your keys. Here's why: %s", err.response['Error']['Message'])

    def describe_key(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Describes a key.\n        '
        key_id = input('Enter a key ID or ARN here to get information about the key: ')
        if key_id:
            try:
                key = self.kms_client.describe_key(KeyId=key_id)['KeyMetadata']
            except ClientError as err:
                logging.error("Couldn't get key '%s'. Here's why: %s", key_id, err.response['Error']['Message'])
            else:
                print(f'Got key {key_id}:')
                pprint(key)
        return key_id

    def generate_data_key(self, key_id):
        if False:
            while True:
                i = 10
        '\n        Generates a symmetric data key that can be used for client-side encryption.\n        '
        answer = input(f'Do you want to generate a symmetric data key from key {key_id} (y/n)? ')
        if answer.lower() == 'y':
            try:
                data_key = self.kms_client.generate_data_key(KeyId=key_id, KeySpec='AES_256')
            except ClientError as err:
                logger.error("Couldn't generate a data key for key %s. Here's why: %s", key_id, err.response['Error']['Message'])
            else:
                pprint(data_key)

    def enable_disable_key(self, key_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disables and then enables a key. Gets the key state after each state change.\n        '
        answer = input('Do you want to disable and then enable that key (y/n)? ')
        if answer.lower() == 'y':
            try:
                self.kms_client.disable_key(KeyId=key_id)
                key = self.kms_client.describe_key(KeyId=key_id)['KeyMetadata']
            except ClientError as err:
                logging.error("Couldn't disable key '%s'. Here's why: %s", key_id, err.response['Error']['Message'])
            else:
                print(f"AWS KMS says your key state is: {key['KeyState']}.")
            try:
                self.kms_client.enable_key(KeyId=key_id)
                key = self.kms_client.describe_key(KeyId=key_id)['KeyMetadata']
            except ClientError as err:
                logging.error("Couldn't enable key '%s'. Here's why: %s", key_id, err.response['Error']['Message'])
            else:
                print(f"AWS KMS says your key state is: {key['KeyState']}.")

    def delete_keys(self, keys):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes a list of keys.\n\n        :param keys: The list of keys to delete.\n        '
        answer = input('Do you want to delete these keys (y/n)? ')
        if answer.lower() == 'y':
            window = 7
            for key in keys:
                try:
                    self.kms_client.schedule_key_deletion(KeyId=key['KeyId'], PendingWindowInDays=window)
                except ClientError as err:
                    logging.error("Couldn't delete key %s. Here's why: %s", key['KeyId'], err.response['Error']['Message'])
                else:
                    print(f"Key {key['KeyId']} scheduled for deletion in {window} days.")

def key_management(kms_client):
    if False:
        i = 10
        return i + 15
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Welcome to the AWS Key Management Service (AWS KMS) key management demo.')
    print('-' * 88)
    key_manager = KeyManager(kms_client)
    key_manager.create_key()
    print('-' * 88)
    key_manager.list_keys()
    print('-' * 88)
    key_id = key_manager.describe_key()
    if key_id:
        key_manager.enable_disable_key(key_id)
        print('-' * 88)
        key_manager.generate_data_key(key_id)
    print('-' * 88)
    print('For this demo, we created these keys:')
    for key in key_manager.created_keys:
        print(f"\tKeyId: {key['KeyId']}")
        print(f"\tDescription: {key['Description']}")
        print('-' * 66)
    key_manager.delete_keys(key_manager.created_keys)
    print('\nThanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    try:
        key_management(boto3.client('kms'))
    except Exception:
        logging.exception('Something went wrong with the demo!')