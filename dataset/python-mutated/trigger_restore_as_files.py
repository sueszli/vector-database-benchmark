from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python trigger_restore_as_files.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='04cf684a-d41f-4550-9f70-7708a3a2283b')
    response = client.backup_instances.begin_trigger_restore(resource_group_name='000pikumar', vault_name='PrivatePreviewVault1', backup_instance_name='testInstance1', parameters={'objectType': 'AzureBackupRecoveryPointBasedRestoreRequest', 'recoveryPointId': 'hardcodedRP', 'restoreTargetInfo': {'objectType': 'RestoreFilesTargetInfo', 'recoveryOption': 'FailIfExists', 'restoreLocation': 'southeastasia', 'targetDetails': {'filePrefix': 'restoredblob', 'restoreTargetLocationType': 'AzureBlobs', 'url': 'https://teststorage.blob.core.windows.net/restoretest'}}, 'sourceDataStoreType': 'VaultStore', 'sourceResourceId': '/subscriptions/f75d8d8b-6735-4697-82e1-1a7a3ff0d5d4/resourceGroups/viveksipgtest/providers/Microsoft.DBforPostgreSQL/servers/viveksipgtest/databases/testdb'}).result()
    print(response)
if __name__ == '__main__':
    main()