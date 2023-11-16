from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerregistry\n# USAGE\n    python scope_map_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerRegistryManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.scope_maps.begin_update(resource_group_name='myResourceGroup', registry_name='myRegistry', scope_map_name='myScopeMap', scope_map_update_parameters={'properties': {'actions': ['repositories/myrepository/contentWrite', 'repositories/myrepository/contentRead'], 'description': 'Developer Scopes'}}).result()
    print(response)
if __name__ == '__main__':
    main()