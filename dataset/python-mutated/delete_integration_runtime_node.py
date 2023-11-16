from azure.identity import DefaultAzureCredential
from azure.mgmt.datamigration import DataMigrationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datamigration\n# USAGE\n    python delete_integration_runtime_node.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataMigrationManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-1111-2222-3333-444444444444')
    response = client.sql_migration_services.delete_node(resource_group_name='testrg', sql_migration_service_name='service1', parameters={'integrationRuntimeName': 'IRName', 'nodeName': 'nodeName'})
    print(response)
if __name__ == '__main__':
    main()