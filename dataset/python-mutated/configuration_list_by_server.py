from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdbforpostgresql import CosmosdbForPostgresqlMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdbforpostgresql\n# USAGE\n    python configuration_list_by_server.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CosmosdbForPostgresqlMgmtClient(credential=DefaultAzureCredential(), subscription_id='ffffffff-ffff-ffff-ffff-ffffffffffff')
    response = client.configurations.list_by_server(resource_group_name='TestResourceGroup', cluster_name='testcluster', server_name='testserver')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()