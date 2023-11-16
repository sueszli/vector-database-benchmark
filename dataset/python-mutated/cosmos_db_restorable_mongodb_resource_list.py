from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_restorable_mongodb_resource_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='2296c272-5d55-40d9-bc05-4d56dc2d7588')
    response = client.restorable_mongodb_resources.list(location='WestUS', instance_id='d9b26648-2f53-4541-b3d8-3044f4f9810d')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()