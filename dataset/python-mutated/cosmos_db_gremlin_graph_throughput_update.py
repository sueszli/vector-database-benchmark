from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_gremlin_graph_throughput_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.gremlin_resources.begin_update_gremlin_graph_throughput(resource_group_name='rg1', account_name='ddb1', database_name='databaseName', graph_name='graphName', update_throughput_parameters={'location': 'West US', 'properties': {'resource': {'throughput': 400}}, 'tags': {}}).result()
    print(response)
if __name__ == '__main__':
    main()