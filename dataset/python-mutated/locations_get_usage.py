from azure.identity import DefaultAzureCredential
from azure.mgmt.datalake.store import DataLakeStoreAccountManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datalake-store\n# USAGE\n    python locations_get_usage.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataLakeStoreAccountManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.locations.get_usage(location='WestUS')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()