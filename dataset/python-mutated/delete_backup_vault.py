from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python delete_backup_vault.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='0b352192-dcac-4cc7-992e-a96190ccc68c')
    client.backup_vaults.begin_delete(resource_group_name='SampleResourceGroup', vault_name='swaggerExample').result()
if __name__ == '__main__':
    main()