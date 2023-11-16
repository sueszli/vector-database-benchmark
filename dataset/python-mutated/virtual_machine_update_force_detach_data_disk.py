from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_update_force_detach_data_disk.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_update(resource_group_name='myResourceGroup', vm_name='myVM', parameters={'properties': {'hardwareProfile': {'vmSize': 'Standard_D2_v2'}, 'networkProfile': {'networkInterfaces': [{'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/networkInterfaces/{existing-nic-name}', 'properties': {'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerName': 'myVM'}, 'storageProfile': {'dataDisks': [{'createOption': 'Empty', 'detachOption': 'ForceDetach', 'diskSizeGB': 1023, 'lun': 0, 'toBeDetached': True}, {'createOption': 'Empty', 'diskSizeGB': 1023, 'lun': 1, 'toBeDetached': False}], 'imageReference': {'offer': 'WindowsServer', 'publisher': 'MicrosoftWindowsServer', 'sku': '2016-Datacenter', 'version': 'latest'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'managedDisk': {'storageAccountType': 'Standard_LRS'}, 'name': 'myVMosdisk'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()