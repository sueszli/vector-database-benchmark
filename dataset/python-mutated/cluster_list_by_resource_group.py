from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdbforpostgresql import CosmosdbForPostgresqlMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdbforpostgresql\n# USAGE\n    python cluster_list_by_resource_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CosmosdbForPostgresqlMgmtClient(credential=DefaultAzureCredential(), subscription_id='ffffffff-ffff-ffff-ffff-ffffffffffff')
    response = client.clusters.list_by_resource_group(resource_group_name='TestGroup')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()