from azure.identity import DefaultAzureCredential
from azure.mgmt.azurestackhci import AzureStackHCIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-azurestackhci\n# USAGE\n    python delete_virtual_hard_disk.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AzureStackHCIClient(credential=DefaultAzureCredential(), subscription_id='fd3c3665-1729-4b7b-9a38-238e83b0f98b')
    client.virtual_hard_disks.begin_delete(resource_group_name='test-rg', virtual_hard_disk_name='test-vhd').result()
if __name__ == '__main__':
    main()