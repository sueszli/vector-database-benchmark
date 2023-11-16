import json
from typing import Optional
from botocore.client import ClientError
from pydantic import BaseModel
from prowler.lib.logger import logger
from prowler.lib.scan_filters.scan_filters import is_resource_filtered
from prowler.providers.aws.lib.service.service import AWSService

class Glacier(AWSService):

    def __init__(self, audit_info):
        if False:
            i = 10
            return i + 15
        super().__init__(__class__.__name__, audit_info)
        self.vaults = {}
        self.__threading_call__(self.__list_vaults__)
        self.__threading_call__(self.__get_vault_access_policy__)
        self.__list_tags_for_vault__()

    def __list_vaults__(self, regional_client):
        if False:
            while True:
                i = 10
        logger.info('Glacier - Listing Vaults...')
        try:
            list_vaults_paginator = regional_client.get_paginator('list_vaults')
            for page in list_vaults_paginator.paginate():
                for vault in page['VaultList']:
                    if not self.audit_resources or is_resource_filtered(vault['VaultARN'], self.audit_resources):
                        vault_name = vault['VaultName']
                        vault_arn = vault['VaultARN']
                        self.vaults[vault_arn] = Vault(name=vault_name, arn=vault_arn, region=regional_client.region)
        except Exception as error:
            logger.error(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')

    def __get_vault_access_policy__(self, regional_client):
        if False:
            for i in range(10):
                print('nop')
        logger.info('Glacier - Getting Vault Access Policy...')
        try:
            for vault in self.vaults.values():
                if vault.region == regional_client.region:
                    try:
                        vault_access_policy = regional_client.get_vault_access_policy(vaultName=vault.name)
                        self.vaults[vault.arn].access_policy = json.loads(vault_access_policy['policy']['Policy'])
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'ResourceNotFoundException':
                            self.vaults[vault.arn].access_policy = {}
        except Exception as error:
            logger.error(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')

    def __list_tags_for_vault__(self):
        if False:
            while True:
                i = 10
        logger.info('Glacier - List Tags...')
        try:
            for vault in self.vaults.values():
                regional_client = self.regional_clients[vault.region]
                response = regional_client.list_tags_for_vault(vaultName=vault.name)['Tags']
                vault.tags = [response]
        except Exception as error:
            logger.error(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')

class Vault(BaseModel):
    name: str
    arn: str
    region: str
    access_policy: dict = {}
    tags: Optional[list] = []