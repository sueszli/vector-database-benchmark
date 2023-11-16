from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python find_restorable_time_ranges.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='04cf684a-d41f-4550-9f70-7708a3a2283b')
    response = client.restorable_time_ranges.find(resource_group_name='Blob-Backup', vault_name='ZBlobBackupVaultBVTD3', backup_instance_name='zblobbackuptestsa58', parameters={'endTime': '2021-02-24T00:35:17.6829685Z', 'sourceDataStoreType': 'OperationalStore', 'startTime': '2020-10-17T23:28:17.6829685Z'})
    print(response)
if __name__ == '__main__':
    main()