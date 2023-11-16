from azure.identity import DefaultAzureCredential
from azure.mgmt.batch import BatchManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-batch\n# USAGE\n    python location_list_virtual_machine_skus.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = BatchManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.location.list_supported_virtual_machine_skus(location_name='japaneast')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()