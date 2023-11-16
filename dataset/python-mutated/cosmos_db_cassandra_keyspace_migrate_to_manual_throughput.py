from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_cassandra_keyspace_migrate_to_manual_throughput.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.cassandra_resources.begin_migrate_cassandra_keyspace_to_manual_throughput(resource_group_name='rg1', account_name='ddb1', keyspace_name='keyspaceName').result()
    print(response)
if __name__ == '__main__':
    main()