from azure.identity import DefaultAzureCredential
from azure.mgmt.appconfiguration import AppConfigurationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appconfiguration\n# USAGE\n    python configuration_stores_regenerate_key.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AppConfigurationManagementClient(credential=DefaultAzureCredential(), subscription_id='c80fb759-c965-4c6a-9110-9b2b2d038882')
    response = client.configuration_stores.regenerate_key(resource_group_name='myResourceGroup', config_store_name='contoso', regenerate_key_parameters={'id': '439AD01B4BE67DB1'})
    print(response)
if __name__ == '__main__':
    main()