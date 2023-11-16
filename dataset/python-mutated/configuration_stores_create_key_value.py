from azure.identity import DefaultAzureCredential
from azure.mgmt.appconfiguration import AppConfigurationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appconfiguration\n# USAGE\n    python configuration_stores_create_key_value.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AppConfigurationManagementClient(credential=DefaultAzureCredential(), subscription_id='c80fb759-c965-4c6a-9110-9b2b2d038882')
    response = client.key_values.create_or_update(resource_group_name='myResourceGroup', config_store_name='contoso', key_value_name='myKey$myLabel')
    print(response)
if __name__ == '__main__':
    main()