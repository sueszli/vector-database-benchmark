from azure.identity import DefaultAzureCredential
from azure.mgmt.databox import DataBoxManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databox\n# USAGE\n    python jobs_patch_cmk.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = DataBoxManagementClient(credential=DefaultAzureCredential(), subscription_id='YourSubscriptionId')
    response = client.jobs.begin_update(resource_group_name='YourResourceGroupName', job_name='TestJobName1', job_resource_update_parameter={'properties': {'details': {'keyEncryptionKey': {'kekType': 'CustomerManaged', 'kekUrl': 'https://xxx.xxx.xx', 'kekVaultResourceID': '/subscriptions/YourSubscriptionId/resourceGroups/YourResourceGroupName/providers/Microsoft.KeyVault/vaults/YourKeyVaultName'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()