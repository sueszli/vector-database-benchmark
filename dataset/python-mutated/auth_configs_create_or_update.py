from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python auth_configs_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='651f8027-33e8-4ec4-97b4-f6e9f3dc8744')
    response = client.container_apps_auth_configs.create_or_update(resource_group_name='workerapps-rg-xj', container_app_name='testcanadacentral', auth_config_name='current', auth_config_envelope={'properties': {'globalValidation': {'unauthenticatedClientAction': 'AllowAnonymous'}, 'identityProviders': {'facebook': {'registration': {'appId': '123', 'appSecretSettingName': 'facebook-secret'}}}, 'platform': {'enabled': True}}})
    print(response)
if __name__ == '__main__':
    main()