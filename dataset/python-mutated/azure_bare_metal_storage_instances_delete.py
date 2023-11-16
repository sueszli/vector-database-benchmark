from azure.identity import DefaultAzureCredential
from azure.mgmt.baremetalinfrastructure import BareMetalInfrastructureClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-baremetalinfrastructure\n# USAGE\n    python azure_bare_metal_storage_instances_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = BareMetalInfrastructureClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    client.azure_bare_metal_storage_instances.delete(resource_group_name='myResourceGroup', azure_bare_metal_storage_instance_name='myAzureBareMetalStorageInstance')
if __name__ == '__main__':
    main()