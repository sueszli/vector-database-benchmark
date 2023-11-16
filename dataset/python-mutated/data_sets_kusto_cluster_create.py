from azure.identity import DefaultAzureCredential
from azure.mgmt.datashare import DataShareManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datashare\n# USAGE\n    python data_sets_kusto_cluster_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = DataShareManagementClient(credential=DefaultAzureCredential(), subscription_id='433a8dfd-e5d5-4e77-ad86-90acdc75eb1a')
    response = client.data_sets.create(resource_group_name='SampleResourceGroup', account_name='Account1', share_name='Share1', data_set_name='Dataset1', data_set={'kind': 'KustoCluster', 'properties': {'kustoClusterResourceId': '/subscriptions/433a8dfd-e5d5-4e77-ad86-90acdc75eb1a/resourceGroups/SampleResourceGroup/providers/Microsoft.Kusto/clusters/Cluster1'}})
    print(response)
if __name__ == '__main__':
    main()