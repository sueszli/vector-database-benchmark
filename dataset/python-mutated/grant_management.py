"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Key Management Service (AWS KMS)
to manage permission grants for keys.
"""
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class GrantManager:

    def __init__(self, kms_client):
        if False:
            print('Hello World!')
        self.kms_client = kms_client

    def create_grant(self, key_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a grant for a key that lets a principal generate a symmetric data\n        encryption key.\n\n        :param key_id: The ARN or ID of the key.\n        :return: The grant that is created.\n        '
        principal = input(f'Enter the ARN of a principal, such as an IAM role, to grant that role GenerateDataKey permissions on key {key_id}: ')
        if principal != '':
            try:
                grant = self.kms_client.create_grant(KeyId=key_id, GranteePrincipal=principal, Operations=['GenerateDataKey'])
            except ClientError as err:
                logger.error("Couldn't create a grant on key %s. Here's why: %s", key_id, err.response['Error']['Message'])
            else:
                print(f'Grant created on key {key_id}.')
                return grant
        else:
            print('Skipping grant creation.')

    def list_grants(self, key_id):
        if False:
            print('Hello World!')
        '\n        Lists grants for a key.\n\n        :param key_id: The ARN or ID of the key to query.\n        :return: The grants for the key.\n        '
        answer = input(f'Ready to list grants on key {key_id} (y/n)? ')
        if answer.lower() == 'y':
            try:
                grants = self.kms_client.list_grants(KeyId=key_id)['Grants']
            except ClientError as err:
                logger.error("Couldn't list grants for key %s. Here's why: %s", key_id, err.response['Error']['Message'])
            else:
                print(f'Grants for key {key_id}:')
                pprint(grants)
                return grants

    def retire_grant(self, grant):
        if False:
            while True:
                i = 10
        '\n        Retires a grant so that it can no longer be used.\n\n        :param grant: The grant to retire.\n        '
        try:
            self.kms_client.retire_grant(GrantToken=grant['GrantToken'])
        except ClientError as err:
            logger.error("Couldn't retire grant %s. Here's why: %s", grant['GrantId'], err.response['Error']['Message'])
        else:
            print(f"Grant {grant['GrantId']} retired.")

    def revoke_grant(self, key_id, grant):
        if False:
            i = 10
            return i + 15
        '\n        Revokes a grant so that it can no longer be used.\n\n        :param key_id: The ARN or ID of the key associated with the grant.\n        :param grant: The grant to revoke.\n        '
        try:
            self.kms_client.revoke_grant(KeyId=key_id, GrantId=grant['GrantId'])
        except ClientError as err:
            logger.error("Couldn't revoke grant %s. Here's why: %s", grant['GrantId'], err.response['Error']['Message'])
        else:
            print(f"Grant {grant['GrantId']} revoked.")

def grant_management(kms_client):
    if False:
        i = 10
        return i + 15
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Welcome to the AWS Key Management Service (AWS KMS) grant management demo.')
    print('-' * 88)
    key_id = input('Enter a key ID or ARN to start the demo: ')
    if key_id == '':
        print('A key is required to run this demo.')
        return
    grant_manager = GrantManager(kms_client)
    grant = grant_manager.create_grant(key_id)
    print('-' * 88)
    grant_manager.list_grants(key_id)
    print('-' * 88)
    if grant is not None:
        action = input("Let's remove the demo grant. Enter 'retire' or 'revoke': ")
        if action == 'retire':
            grant_manager.retire_grant(grant)
        elif action == 'revoke':
            grant_manager.revoke_grant(key_id, grant)
        else:
            print('Skipping grant removal.')
    print('\nThanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    try:
        grant_management(boto3.client('kms'))
    except Exception:
        logging.exception('Something went wrong with the demo!')