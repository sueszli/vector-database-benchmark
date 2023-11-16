from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python bindings_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.bindings.begin_create_or_update(resource_group_name='myResourceGroup', service_name='myservice', app_name='myapp', binding_name='mybinding', binding_resource={'properties': {'bindingParameters': {'apiType': 'SQL', 'databaseName': 'db1'}, 'createdAt': None, 'generatedProperties': None, 'key': 'xxxx', 'resourceId': '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroup/providers/Microsoft.DocumentDB/databaseAccounts/my-cosmosdb-1', 'updatedAt': None}}).result()
    print(response)
if __name__ == '__main__':
    main()