from azure.identity import DefaultAzureCredential
from azure.mgmt.datashare import DataShareManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datashare\n# USAGE\n    python data_sets_synapse_workspace_sql_pool_table_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataShareManagementClient(credential=DefaultAzureCredential(), subscription_id='0f3dcfc3-18f8-4099-b381-8353e19d43a7')
    response = client.data_sets.create(resource_group_name='SampleResourceGroup', account_name='sourceAccount', share_name='share1', data_set_name='dataset1', data_set={'kind': 'SynapseWorkspaceSqlPoolTable', 'properties': {'synapseWorkspaceSqlPoolTableResourceId': '/subscriptions/0f3dcfc3-18f8-4099-b381-8353e19d43a7/resourceGroups/SampleResourceGroup/providers/Microsoft.Synapse/workspaces/ExampleWorkspace/sqlPools/ExampleSqlPool/schemas/dbo/tables/table1'}})
    print(response)
if __name__ == '__main__':
    main()