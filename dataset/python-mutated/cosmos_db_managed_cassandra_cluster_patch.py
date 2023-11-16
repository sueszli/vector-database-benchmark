from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_managed_cassandra_cluster_patch.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.cassandra_clusters.begin_update(resource_group_name='cassandra-prod-rg', cluster_name='cassandra-prod', body={'properties': {'authenticationMethod': 'None', 'externalGossipCertificates': [{'pem': '-----BEGIN CERTIFICATE-----\n...Base64 encoded certificate...\n-----END CERTIFICATE-----'}], 'externalSeedNodes': [{'ipAddress': '10.52.221.2'}, {'ipAddress': '10.52.221.3'}, {'ipAddress': '10.52.221.4'}], 'hoursBetweenBackups': 12}, 'tags': {'owner': 'mike'}}).result()
    print(response)
if __name__ == '__main__':
    main()