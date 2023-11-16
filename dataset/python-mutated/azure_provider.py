import sys
from os import getenv
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.mgmt.subscription import SubscriptionClient
from msgraph.core import GraphClient
from prowler.lib.logger import logger
from prowler.providers.azure.lib.audit_info.models import Azure_Identity_Info
from prowler.providers.azure.lib.regions.regions import get_regions_config

class Azure_Provider:

    def __init__(self, az_cli_auth: bool, sp_env_auth: bool, browser_auth: bool, managed_entity_auth: bool, subscription_ids: list, tenant_id: str, region: str):
        if False:
            for i in range(10):
                print('nop')
        logger.info('Instantiating Azure Provider ...')
        self.region_config = self.__get_region_config__(region)
        self.credentials = self.__get_credentials__(az_cli_auth, sp_env_auth, browser_auth, managed_entity_auth, tenant_id)
        self.identity = self.__get_identity_info__(self.credentials, az_cli_auth, sp_env_auth, browser_auth, managed_entity_auth, subscription_ids)

    def __get_region_config__(self, region):
        if False:
            print('Hello World!')
        return get_regions_config(region)

    def __get_credentials__(self, az_cli_auth, sp_env_auth, browser_auth, managed_entity_auth, tenant_id):
        if False:
            return 10
        if not browser_auth:
            if sp_env_auth:
                self.__check_sp_creds_env_vars__()
            try:
                credentials = DefaultAzureCredential(exclude_environment_credential=not sp_env_auth, exclude_cli_credential=not az_cli_auth, exclude_managed_identity_credential=not managed_entity_auth, exclude_visual_studio_code_credential=True, exclude_shared_token_cache_credential=True, exclude_powershell_credential=True, authority=self.region_config['authority'])
            except Exception as error:
                logger.critical('Failed to retrieve azure credentials')
                logger.critical(f'{error.__class__.__name__}[{error.__traceback__.tb_lineno}] -- {error}')
                sys.exit(1)
        else:
            try:
                credentials = InteractiveBrowserCredential(tenant_id=tenant_id)
            except Exception as error:
                logger.critical('Failed to retrieve azure credentials')
                logger.critical(f'{error.__class__.__name__}[{error.__traceback__.tb_lineno}] -- {error}')
                sys.exit(1)
        return credentials

    def __check_sp_creds_env_vars__(self):
        if False:
            while True:
                i = 10
        logger.info('Azure provider: checking service principal environment variables  ...')
        for env_var in ['AZURE_CLIENT_ID', 'AZURE_TENANT_ID', 'AZURE_CLIENT_SECRET']:
            if not getenv(env_var):
                logger.critical(f'Azure provider: Missing environment variable {env_var} needed to autenticate against Azure')
                sys.exit(1)

    def __get_identity_info__(self, credentials, az_cli_auth, sp_env_auth, browser_auth, managed_entity_auth, subscription_ids):
        if False:
            return 10
        identity = Azure_Identity_Info()
        if sp_env_auth or browser_auth or az_cli_auth:
            try:
                logger.info('Trying to retrieve tenant domain from AAD to populate identity structure ...')
                client = GraphClient(credential=credentials)
                domain_result = client.get('/domains').json()
                if 'value' in domain_result:
                    if 'id' in domain_result['value'][0]:
                        identity.domain = domain_result['value'][0]['id']
            except Exception as error:
                logger.error('Provided identity does not have permissions to access AAD to retrieve tenant domain')
                logger.error(f'{error.__class__.__name__}[{error.__traceback__.tb_lineno}] -- {error}')
            if sp_env_auth:
                identity.identity_id = getenv('AZURE_CLIENT_ID')
                identity.identity_type = 'Service Principal'
            else:
                identity.identity_id = 'Unknown user id (Missing AAD permissions)'
                identity.identity_type = 'User'
                try:
                    logger.info('Trying to retrieve user information from AAD to populate identity structure ...')
                    client = GraphClient(credential=credentials)
                    user_name = client.get('/me').json()
                    if 'userPrincipalName' in user_name:
                        identity.identity_id = user_name
                except Exception as error:
                    logger.error("Provided identity does not have permissions to access AAD to retrieve user's metadata")
                    logger.error(f'{error.__class__.__name__}[{error.__traceback__.tb_lineno}] -- {error}')
        elif managed_entity_auth:
            identity.identity_id = 'Default Managed Identity ID'
            identity.identity_type = 'Managed Identity'
        try:
            logger.info('Trying to subscriptions and tenant ids to populate identity structure ...')
            subscriptions_client = SubscriptionClient(credential=credentials, base_url=self.region_config['base_url'], credential_scopes=self.region_config['credential_scopes'])
            if not subscription_ids:
                logger.info('Scanning all the Azure subscriptions...')
                for subscription in subscriptions_client.subscriptions.list():
                    identity.subscriptions.update({subscription.display_name: subscription.subscription_id})
            else:
                logger.info('Scanning the subscriptions passed as argument ...')
                for id in subscription_ids:
                    subscription = subscriptions_client.subscriptions.get(subscription_id=id)
                    identity.subscriptions.update({subscription.display_name: id})
            if not identity.subscriptions:
                logger.critical('It was not possible to retrieve any subscriptions, please check your permission assignments')
                sys.exit(1)
            tenants = subscriptions_client.tenants.list()
            for tenant in tenants:
                identity.tenant_ids.append(tenant.tenant_id)
        except Exception as error:
            logger.critical('Error with credentials provided getting subscriptions and tenants to scan')
            logger.critical(f'{error.__class__.__name__}[{error.__traceback__.tb_lineno}] -- {error}')
            sys.exit(1)
        return identity

    def get_credentials(self):
        if False:
            print('Hello World!')
        return self.credentials

    def get_identity(self):
        if False:
            return 10
        return self.identity

    def get_region_config(self):
        if False:
            return 10
        return self.region_config