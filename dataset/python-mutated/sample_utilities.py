"""
FILE: sample_utilities.py
DESCRIPTION:
    This file include some utility functions for samples to use:
    - get_authority(): get authority of the ConfigurationClient
    - get_audience(): get audience of the ConfigurationClient
    - get_credential(): get credential of the ConfigurationClient
    It is not a file expected to run independently.
"""
import os
from azure.identity import AzureAuthorityHosts, ClientSecretCredential
from azure.identity.aio import ClientSecretCredential as AsyncClientSecretCredential

def get_authority(endpoint):
    if False:
        return 10
    if '.azconfig.io' in endpoint:
        return AzureAuthorityHosts.AZURE_PUBLIC_CLOUD
    if '.azconfig.azure.cn' in endpoint:
        return AzureAuthorityHosts.AZURE_CHINA
    if '.azconfig.azure.us' in endpoint:
        return AzureAuthorityHosts.AZURE_GOVERNMENT
    raise ValueError(f'Endpoint ({endpoint}) could not be understood')

def get_audience(authority):
    if False:
        i = 10
        return i + 15
    if authority == AzureAuthorityHosts.AZURE_PUBLIC_CLOUD:
        return 'https://management.azure.com'
    if authority == AzureAuthorityHosts.AZURE_CHINA:
        return 'https://management.chinacloudapi.cn'
    if authority == AzureAuthorityHosts.AZURE_GOVERNMENT:
        return 'https://management.usgovcloudapi.net'

def get_credential(authority, **kwargs):
    if False:
        print('Hello World!')
    if kwargs.pop('is_async', False):
        return AsyncClientSecretCredential(tenant_id=os.environ.get('APPCONFIGURATION_TENANT_ID'), client_id=os.environ.get('APPCONFIGURATION_CLIENT_ID'), client_secret=os.environ.get('APPCONFIGURATION_CLIENT_SECRET'), authority=authority)
    return ClientSecretCredential(tenant_id=os.environ.get('APPCONFIGURATION_TENANT_ID'), client_id=os.environ.get('APPCONFIGURATION_CLIENT_ID'), client_secret=os.environ.get('APPCONFIGURATION_CLIENT_SECRET'), authority=authority)

def get_client_modifications():
    if False:
        for i in range(10):
            print('nop')
    modifications = {}
    modifications['user_agent'] = 'SDK/Sample'
    return modifications