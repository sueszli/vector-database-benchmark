from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python undelete_deleted_backup_instance.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='04cf684a-d41f-4550-9f70-7708a3a2283b')
    client.deleted_backup_instances.begin_undelete(resource_group_name='testrg', vault_name='testvault', backup_instance_name='testbi').result()
if __name__ == '__main__':
    main()