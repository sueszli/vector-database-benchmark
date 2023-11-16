from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_scale_set_vm_run_command_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machine_scale_set_vm_run_commands.begin_update(resource_group_name='myResourceGroup', vm_scale_set_name='myvmScaleSet', instance_id='0', run_command_name='myRunCommand', run_command={'properties': {'source': {'scriptUri': 'https://mystorageaccount.blob.core.windows.net/scriptcontainer/MyScript.ps1', 'scriptUriManagedIdentity': {'objectId': '4231e4d2-33e4-4e23-96b2-17888afa6072'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()