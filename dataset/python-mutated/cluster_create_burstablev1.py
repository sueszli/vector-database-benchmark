from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdbforpostgresql import CosmosdbForPostgresqlMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdbforpostgresql\n# USAGE\n    python cluster_create_burstablev1.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CosmosdbForPostgresqlMgmtClient(credential=DefaultAzureCredential(), subscription_id='ffffffff-ffff-ffff-ffff-ffffffffffff')
    response = client.clusters.begin_create(resource_group_name='TestGroup', cluster_name='testcluster-burstablev1', parameters={'location': 'westus', 'properties': {'administratorLoginPassword': 'password', 'citusVersion': '11.3', 'coordinatorEnablePublicIpAccess': True, 'coordinatorServerEdition': 'BurstableMemoryOptimized', 'coordinatorStorageQuotaInMb': 131072, 'coordinatorVCores': 1, 'enableHa': False, 'enableShardsOnCoordinator': True, 'nodeCount': 0, 'postgresqlVersion': '15', 'preferredPrimaryZone': '1'}, 'tags': {'owner': 'JohnDoe'}}).result()
    print(response)
if __name__ == '__main__':
    main()