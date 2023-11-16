"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Key Management Service (AWS KMS)
to manage key aliases.
"""
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class AliasManager:

    def __init__(self, kms_client):
        if False:
            i = 10
            return i + 15
        self.kms_client = kms_client
        self.created_key = None

    def setup(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets up a key for the demo. Either creates a new key or uses one supplied by\n        the user.\n\n        :return: The ARN or ID of the key to use for the demo.\n        '
        answer = input('Do you want to create a new key for the demo (y/n)? ')
        if answer.lower() == 'y':
            try:
                key = self.kms_client.create_key(Description='Alias management demo key')['KeyMetadata']
                self.created_key = key
            except ClientError as err:
                logger.error("Couldn't create key. Here's why: %s", err.response['Error']['Message'])
                raise
            else:
                key_id = key['KeyId']
        else:
            key_id = input('Enter a key ID or ARN to use for the demo: ')
            if key_id == '':
                key_id = None
        return key_id

    def teardown(self):
        if False:
            i = 10
            return i + 15
        '\n        Deletes any resources that were created for the demo.\n        '
        if self.created_key is not None:
            answer = input(f"Key {self.created_key['KeyId']} was created for this demo. Do you want to delete it (y/n)? ")
            if answer.lower() == 'y':
                try:
                    self.kms_client.schedule_key_deletion(KeyId=self.created_key['KeyId'], PendingWindowInDays=7)
                except ClientError as err:
                    logging.error("Couldn't delete key. Here's why: %s", err.response['Error']['Message'])
                else:
                    print(f'Key scheduled for deletion in 7 days.')

    def create_alias(self, key_id):
        if False:
            while True:
                i = 10
        '\n        Creates an alias for the specified key.\n\n        :param key_id: The ARN or ID of a key to give an alias.\n        :return: The alias given to the key.\n        '
        alias = ''
        while alias == '':
            alias = input(f'What alias would you like to give to key {key_id}? ')
        try:
            self.kms_client.create_alias(AliasName=alias, TargetKeyId=key_id)
        except ClientError as err:
            logger.error("Couldn't create alias %s. Here's why: %s", alias, err.response['Error']['Message'])
        else:
            print(f'Created alias {alias} for key {key_id}.')
            return alias

    def list_aliases(self):
        if False:
            i = 10
            return i + 15
        '\n        Lists aliases for the current account.\n        '
        answer = input("\nLet's list your key aliases. Ready (y/n)? ")
        if answer.lower() == 'y':
            try:
                page_size = 10
                alias_paginator = self.kms_client.get_paginator('list_aliases')
                for alias_page in alias_paginator.paginate(PaginationConfig={'PageSize': 10}):
                    print(f'Here are {page_size} aliases:')
                    pprint(alias_page['Aliases'])
                    if alias_page['Truncated']:
                        answer = input(f'Do you want to see the next {page_size} aliases (y/n)? ')
                        if answer.lower() != 'y':
                            break
                    else:
                        print("That's all your aliases!")
            except ClientError as err:
                logging.error("Couldn't list your aliases. Here's why: %s", err.response['Error']['Message'])

    def update_alias(self, alias, current_key_id):
        if False:
            print('Hello World!')
        '\n        Updates an alias by assigning it to another key.\n\n        :param alias: The alias to reassign.\n        :param current_key_id: The ARN or ID of the key currently associated with the alias.\n        '
        new_key_id = input(f'Alias {alias} is currently associated with {current_key_id}. Enter another key ID or ARN that you want to associate with {alias}: ')
        if new_key_id != '':
            try:
                self.kms_client.update_alias(AliasName=alias, TargetKeyId=new_key_id)
            except ClientError as err:
                logger.error("Couldn't associate alias %s with key %s. Here's why: %s", alias, new_key_id, err.response['Error']['Message'])
            else:
                print(f'Alias {alias} is now associated with key {new_key_id}.')
        else:
            print('Skipping alias update.')

    def delete_alias(self):
        if False:
            i = 10
            return i + 15
        '\n        Deletes an alias.\n        '
        alias = input(f"Enter an alias that you'd like to delete: ")
        if alias != '':
            try:
                self.kms_client.delete_alias(AliasName=alias)
            except ClientError as err:
                logger.error("Couldn't delete alias %s. Here's why: %s", alias, err.response['Error']['Message'])
            else:
                print(f'Deleted alias {alias}.')
        else:
            print('Skipping alias deletion.')

def alias_management(kms_client):
    if False:
        for i in range(10):
            print('nop')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Welcome to the AWS Key Management Service (AWS KMS) alias management demo.')
    print('-' * 88)
    alias_manager = AliasManager(kms_client)
    key_id = None
    while key_id is None:
        key_id = alias_manager.setup()
    print('-' * 88)
    alias = None
    while alias is None:
        alias = alias_manager.create_alias(key_id)
    print('-' * 88)
    alias_manager.list_aliases()
    print('-' * 88)
    alias_manager.update_alias(alias, key_id)
    print('-' * 88)
    alias_manager.delete_alias()
    print('-' * 88)
    alias_manager.teardown()
    print('\nThanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    try:
        alias_management(boto3.client('kms'))
    except Exception:
        logging.exception('Something went wrong with the demo!')