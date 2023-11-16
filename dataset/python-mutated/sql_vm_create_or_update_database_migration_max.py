from azure.identity import DefaultAzureCredential
from azure.mgmt.datamigration import DataMigrationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datamigration\n# USAGE\n    python sql_vm_create_or_update_database_migration_max.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataMigrationManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-1111-2222-3333-444444444444')
    response = client.database_migrations_sql_vm.begin_create_or_update(resource_group_name='testrg', sql_virtual_machine_name='testvm', target_db_name='db1', parameters={'properties': {'backupConfiguration': {'sourceLocation': {'fileShare': {'password': 'placeholder', 'path': 'C:\\aaa\\bbb\\ccc', 'username': 'name'}}, 'targetLocation': {'accountKey': 'abcd', 'storageAccountResourceId': 'account.database.windows.net'}}, 'kind': 'SqlVm', 'migrationService': '/subscriptions/00000000-1111-2222-3333-444444444444/resourceGroups/testrg/providers/Microsoft.DataMigration/sqlMigrationServices/testagent', 'offlineConfiguration': {'lastBackupName': 'last_backup_file_name', 'offline': True}, 'scope': '/subscriptions/00000000-1111-2222-3333-444444444444/resourceGroups/testrg/providers/Microsoft.SqlVirtualMachine/sqlVirtualMachines/testvm', 'sourceDatabaseName': 'aaa', 'sourceSqlConnection': {'authentication': 'WindowsAuthentication', 'dataSource': 'aaa', 'encryptConnection': True, 'password': 'placeholder', 'trustServerCertificate': True, 'userName': 'bbb'}}}).result()
    print(response)
if __name__ == '__main__':
    main()