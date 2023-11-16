from azure.identity import DefaultAzureCredential
from azure.mgmt.datamigration import DataMigrationManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-datamigration\n# USAGE\n    python files_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataMigrationManagementClient(credential=DefaultAzureCredential(), subscription_id='fc04246f-04c5-437e-ac5e-206a19e7193f')
    response = client.files.create_or_update(group_name='DmsSdkRg', service_name='DmsSdkService', project_name='DmsSdkProject', file_name='x114d023d8', parameters={'properties': {'filePath': 'DmsSdkFilePath/DmsSdkFile.sql'}})
    print(response)
if __name__ == '__main__':
    main()