from azure.identity import DefaultAzureCredential
from azure.mgmt.datamigration import DataMigrationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datamigration\n# USAGE\n    python service_tasks_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = DataMigrationManagementClient(credential=DefaultAzureCredential(), subscription_id='fc04246f-04c5-437e-ac5e-206a19e7193f')
    response = client.service_tasks.update(group_name='DmsSdkRg', service_name='DmsSdkService', task_name='DmsSdkTask', parameters={'properties': {'input': {'serverVersion': 'NA'}, 'taskType': 'Service.Check.OCI'}})
    print(response)
if __name__ == '__main__':
    main()