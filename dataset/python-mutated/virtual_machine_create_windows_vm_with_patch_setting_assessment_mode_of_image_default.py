from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_create_windows_vm_with_patch_setting_assessment_mode_of_image_default.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_create_or_update(resource_group_name='myResourceGroup', vm_name='myVM', parameters={'location': 'westus', 'properties': {'hardwareProfile': {'vmSize': 'Standard_D1_v2'}, 'networkProfile': {'networkInterfaces': [{'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/networkInterfaces/{existing-nic-name}', 'properties': {'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerName': 'myVM', 'windowsConfiguration': {'enableAutomaticUpdates': True, 'patchSettings': {'assessmentMode': 'ImageDefault'}, 'provisionVMAgent': True}}, 'storageProfile': {'imageReference': {'offer': 'WindowsServer', 'publisher': 'MicrosoftWindowsServer', 'sku': '2016-Datacenter', 'version': 'latest'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'managedDisk': {'storageAccountType': 'Premium_LRS'}, 'name': 'myVMosdisk'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()