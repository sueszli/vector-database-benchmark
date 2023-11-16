from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_extension_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machine_extensions.begin_update(resource_group_name='myResourceGroup', vm_name='myVM', vm_extension_name='myVMExtension', extension_parameters={'properties': {'autoUpgradeMinorVersion': True, 'protectedSettingsFromKeyVault': {'secretUrl': 'https://kvName.vault.azure.net/secrets/secretName/79b88b3a6f5440ffb2e73e44a0db712e', 'sourceVault': {'id': '/subscriptions/a53f7094-a16c-47af-abe4-b05c05d0d79a/resourceGroups/myResourceGroup/providers/Microsoft.KeyVault/vaults/kvName'}}, 'publisher': 'extPublisher', 'settings': {'UserName': 'xyz@microsoft.com'}, 'suppressFailures': True, 'type': 'extType', 'typeHandlerVersion': '1.2'}}).result()
    print(response)
if __name__ == '__main__':
    main()