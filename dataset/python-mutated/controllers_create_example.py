from azure.identity import DefaultAzureCredential
from azure.mgmt.devspaces import DevSpacesManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-devspaces\n# USAGE\n    python controllers_create_example.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DevSpacesManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.controllers.begin_create(resource_group_name='myResourceGroup', name='myControllerResource', controller={'location': 'eastus', 'properties': {'targetContainerHostCredentialsBase64': 'QmFzZTY0IEVuY29kZWQgVmFsdWUK', 'targetContainerHostResourceId': '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroup/providers/Microsoft.ContainerService/managedClusters/myCluster'}, 'sku': {'name': 'S1', 'tier': 'Standard'}, 'tags': {}}).result()
    print(response)
if __name__ == '__main__':
    main()