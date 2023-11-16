from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datafactory\n# USAGE\n    python credentials_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DataFactoryManagementClient(credential=DefaultAzureCredential(), subscription_id='12345678-1234-1234-1234-12345678abc')
    response = client.credential_operations.create_or_update(resource_group_name='exampleResourceGroup', factory_name='exampleFactoryName', credential_name='exampleCredential', credential={'properties': {'type': 'ManagedIdentity', 'typeProperties': {'resourceId': '/subscriptions/12345678-1234-1234-1234-12345678abc/resourcegroups/exampleResourceGroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/exampleUami'}}})
    print(response)
if __name__ == '__main__':
    main()