from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerinstance\n# USAGE\n    python container_group_encryption_properties.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ContainerInstanceManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.container_groups.begin_create_or_update(resource_group_name='demo', container_group_name='demo1', container_group={'identity': {'type': 'UserAssigned', 'userAssignedIdentities': {'/subscriptions/00000000-0000-0000-0000-000000000000/resourcegroups/test-rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/container-group-identity': {}}}, 'location': 'eastus2', 'properties': {'containers': [{'name': 'demo1', 'properties': {'command': [], 'environmentVariables': [], 'image': 'nginx', 'ports': [{'port': 80}], 'resources': {'requests': {'cpu': 1, 'memoryInGB': 1.5}}}}], 'encryptionProperties': {'identity': '/subscriptions/00000000-0000-0000-0000-000000000000/resourcegroups/test-rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/container-group-identity', 'keyName': 'test-key', 'keyVersion': '<key version>', 'vaultBaseUrl': 'https://testkeyvault.vault.azure.net'}, 'imageRegistryCredentials': [], 'ipAddress': {'ports': [{'port': 80, 'protocol': 'TCP'}], 'type': 'Public'}, 'osType': 'Linux'}}).result()
    print(response)
if __name__ == '__main__':
    main()