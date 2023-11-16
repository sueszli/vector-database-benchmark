from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cosmosdb\n# USAGE\n    python cosmos_db_database_account_patch.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CosmosDBManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.database_accounts.begin_update(resource_group_name='rg1', account_name='ddb1', update_parameters={'identity': {'type': 'SystemAssigned,UserAssigned', 'userAssignedIdentities': {'/subscriptions/fa5fc227-a624-475e-b696-cdd604c735bc/resourceGroups/eu2cgroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id1': {}}}, 'location': 'westus', 'properties': {'analyticalStorageConfiguration': {'schemaType': 'WellDefined'}, 'backupPolicy': {'periodicModeProperties': {'backupIntervalInMinutes': 240, 'backupRetentionIntervalInHours': 720, 'backupStorageRedundancy': 'Local'}, 'type': 'Periodic'}, 'capacity': {'totalThroughputLimit': 2000}, 'consistencyPolicy': {'defaultConsistencyLevel': 'BoundedStaleness', 'maxIntervalInSeconds': 10, 'maxStalenessPrefix': 200}, 'defaultIdentity': 'FirstPartyIdentity', 'enableAnalyticalStorage': True, 'enableBurstCapacity': True, 'enableFreeTier': False, 'enablePartitionMerge': True, 'ipRules': [{'ipAddressOrRange': '23.43.230.120'}, {'ipAddressOrRange': '110.12.240.0/12'}], 'isVirtualNetworkFilterEnabled': True, 'minimalTlsVersion': 'Tls', 'networkAclBypass': 'AzureServices', 'networkAclBypassResourceIds': ['/subscriptions/subId/resourcegroups/rgName/providers/Microsoft.Synapse/workspaces/workspaceName'], 'virtualNetworkRules': [{'id': '/subscriptions/subId/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet1/subnets/subnet1', 'ignoreMissingVNetServiceEndpoint': False}]}, 'tags': {'dept': 'finance'}}).result()
    print(response)
if __name__ == '__main__':
    main()