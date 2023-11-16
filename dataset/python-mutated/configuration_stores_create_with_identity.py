from azure.identity import DefaultAzureCredential
from azure.mgmt.appconfiguration import AppConfigurationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appconfiguration\n# USAGE\n    python configuration_stores_create_with_identity.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AppConfigurationManagementClient(credential=DefaultAzureCredential(), subscription_id='c80fb759-c965-4c6a-9110-9b2b2d038882')
    response = client.configuration_stores.begin_create(resource_group_name='myResourceGroup', config_store_name='contoso', config_store_creation_parameters={'identity': {'type': 'SystemAssigned, UserAssigned', 'userAssignedIdentities': {'/subscriptions/c80fb759-c965-4c6a-9110-9b2b2d038882/resourcegroups/myResourceGroup1/providers/Microsoft.ManagedIdentity/userAssignedIdentities/identity2': {}}}, 'location': 'westus', 'sku': {'name': 'Standard'}, 'tags': {'myTag': 'myTagValue'}}).result()
    print(response)
if __name__ == '__main__':
    main()