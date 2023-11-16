from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_mongo_db_database_create_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.mongo_db_resources.begin_create_update_mongo_db_database(resource_group_name='rg1', account_name='ddb1', database_name='databaseName', create_update_mongo_db_database_parameters={'location': 'West US', 'properties': {'options': {}, 'resource': {'id': 'databaseName'}}, 'tags': {}}).result()
    print(response)
if __name__ == '__main__':
    main()