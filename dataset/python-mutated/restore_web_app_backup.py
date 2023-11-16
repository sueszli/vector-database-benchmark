from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python restore_web_app_backup.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.web_apps.begin_restore(resource_group_name='testrg123', name='sitef6141', backup_id='123244', request={'properties': {'databases': [{'connectionString': 'DSN=data-source-name[;SERVER=value] [;PWD=value] [;UID=value] [;<Attribute>=<value>]', 'connectionStringName': 'backend', 'databaseType': 'SqlAzure', 'name': 'backenddb'}, {'connectionString': 'DSN=data-source-name[;SERVER=value] [;PWD=value] [;UID=value] [;<Attribute>=<value>]', 'connectionStringName': 'stats', 'databaseType': 'SqlAzure', 'name': 'statsdb'}], 'overwrite': True, 'siteName': 'sitef6141', 'storageAccountUrl': 'DefaultEndpointsProtocol=https;AccountName=storagesample;AccountKey=<account-key>'}}).result()
    print(response)
if __name__ == '__main__':
    main()