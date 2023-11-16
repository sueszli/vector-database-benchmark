from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datafactory\n# USAGE\n    python approve_reject_private_endpoint_connection.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataFactoryManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.private_endpoint_connection.create_or_update(resource_group_name='exampleResourceGroup', factory_name='exampleFactoryName', private_endpoint_connection_name='connection', private_endpoint_wrapper={'properties': {'privateEndpoint': {'id': '/subscriptions/12345678-1234-1234-1234-12345678abc/resourceGroups/exampleResourceGroup/providers/Microsoft.DataFactory/factories/exampleFactoryName/privateEndpoints/myPrivateEndpoint'}, 'privateLinkServiceConnectionState': {'actionsRequired': '', 'description': 'Approved by admin.', 'status': 'Approved'}}})
    print(response)
if __name__ == '__main__':
    main()