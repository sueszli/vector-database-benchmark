"""Demonstrates how to perform basic operations with Google Cloud IAM
service account keys.

For more information, see the documentation at
https://cloud.google.com/iam/docs/creating-managing-service-account-keys.
"""
import argparse
import os
from google.oauth2 import service_account
import googleapiclient.discovery

def create_key(service_account_email: str) -> None:
    if False:
        print('Hello World!')
    'Creates a key for a service account.'
    credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    service = googleapiclient.discovery.build('iam', 'v1', credentials=credentials)
    key = service.projects().serviceAccounts().keys().create(name='projects/-/serviceAccounts/' + service_account_email, body={}).execute()
    if not key['disabled']:
        print('Created json key')

def list_keys(service_account_email: str) -> None:
    if False:
        return 10
    'Lists all keys for a service account.'
    credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    service = googleapiclient.discovery.build('iam', 'v1', credentials=credentials)
    keys = service.projects().serviceAccounts().keys().list(name='projects/-/serviceAccounts/' + service_account_email).execute()
    for key in keys['keys']:
        print('Key: ' + key['name'])

def delete_key(full_key_name: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes a service account key.'
    credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    service = googleapiclient.discovery.build('iam', 'v1', credentials=credentials)
    service.projects().serviceAccounts().keys().delete(name=full_key_name).execute()
    print('Deleted key: ' + full_key_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    create_key_parser = subparsers.add_parser('create', help=create_key.__doc__)
    create_key_parser.add_argument('service_account_email')
    list_keys_parser = subparsers.add_parser('list', help=list_keys.__doc__)
    list_keys_parser.add_argument('service_account_email')
    delete_key_parser = subparsers.add_parser('delete', help=delete_key.__doc__)
    delete_key_parser.add_argument('full_key_name')
    args = parser.parse_args()
    if args.command == 'list':
        list_keys(args.service_account_email)
    elif args.command == 'create':
        create_key(args.service_account_email)
    elif args.command == 'delete':
        delete_key(args.full_key_name)