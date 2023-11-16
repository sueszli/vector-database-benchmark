from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerregistry\n# USAGE\n    python import_image_from_public_registry.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ContainerRegistryManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    client.registries.begin_import_image(resource_group_name='myResourceGroup', registry_name='myRegistry', parameters={'mode': 'Force', 'source': {'registryUri': 'registry.hub.docker.com', 'sourceImage': 'library/hello-world'}, 'targetTags': ['targetRepository:targetTag'], 'untaggedTargetRepositories': ['targetRepository1']}).result()
if __name__ == '__main__':
    main()