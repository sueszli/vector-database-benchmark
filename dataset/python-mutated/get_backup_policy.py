from azure.identity import DefaultAzureCredential
from azure.mgmt.dataprotection import DataProtectionMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dataprotection\n# USAGE\n    python get_backup_policy.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DataProtectionMgmtClient(credential=DefaultAzureCredential(), subscription_id='04cf684a-d41f-4550-9f70-7708a3a2283b')
    response = client.backup_policies.get(resource_group_name='000pikumar', vault_name='PrivatePreviewVault', backup_policy_name='OSSDBPolicy')
    print(response)
if __name__ == '__main__':
    main()