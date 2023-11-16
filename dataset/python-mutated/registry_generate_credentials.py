from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerregistry\n# USAGE\n    python registry_generate_credentials.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ContainerRegistryManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.registries.begin_generate_credentials(resource_group_name='myResourceGroup', registry_name='myRegistry', generate_credentials_parameters={'expiry': '2020-12-31T15:59:59.0707808Z', 'tokenId': '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroup/providers/Microsoft.ContainerRegistry/registries/myRegistry/tokens/myToken'}).result()
    print(response)
if __name__ == '__main__':
    main()