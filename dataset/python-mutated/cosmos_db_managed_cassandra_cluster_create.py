from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_managed_cassandra_cluster_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.cassandra_clusters.begin_create_update(resource_group_name='cassandra-prod-rg', cluster_name='cassandra-prod', body={'location': 'West US', 'properties': {'authenticationMethod': 'Cassandra', 'cassandraVersion': '3.11', 'clientCertificates': [{'pem': '-----BEGIN CERTIFICATE-----\n...Base64 encoded certificate...\n-----END CERTIFICATE-----'}], 'clusterNameOverride': 'ClusterNameIllegalForAzureResource', 'delegatedManagementSubnetId': '/subscriptions/536e130b-d7d6-4ac7-98a5-de20d69588d2/resourceGroups/customer-vnet-rg/providers/Microsoft.Network/virtualNetworks/customer-vnet/subnets/management', 'externalGossipCertificates': [{'pem': '-----BEGIN CERTIFICATE-----\n...Base64 encoded certificate...\n-----END CERTIFICATE-----'}], 'externalSeedNodes': [{'ipAddress': '10.52.221.2'}, {'ipAddress': '10.52.221.3'}, {'ipAddress': '10.52.221.4'}], 'hoursBetweenBackups': 24, 'initialCassandraAdminPassword': 'mypassword'}, 'tags': {}}).result()
    print(response)
if __name__ == '__main__':
    main()