from azure.identity import DefaultAzureCredential
from azure.mgmt.azurestackhci import AzureStackHCIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-azurestackhci\n# USAGE\n    python get_virtual_machine_instance.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AzureStackHCIClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.virtual_machine_instances.get(resource_uri='subscriptions/fd3c3665-1729-4b7b-9a38-238e83b0f98b/resourceGroups/testrg/Microsoft.HybridCompute/machines/DemoVM')
    print(response)
if __name__ == '__main__':
    main()