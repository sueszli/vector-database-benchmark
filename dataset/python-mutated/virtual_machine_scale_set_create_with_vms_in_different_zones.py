from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_scale_set_create_with_vms_in_different_zones.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machine_scale_sets.begin_create_or_update(resource_group_name='myResourceGroup', vm_scale_set_name='{vmss-name}', parameters={'location': 'centralus', 'properties': {'overprovision': True, 'upgradePolicy': {'mode': 'Automatic'}, 'virtualMachineProfile': {'networkProfile': {'networkInterfaceConfigurations': [{'name': '{vmss-name}', 'properties': {'enableIPForwarding': True, 'ipConfigurations': [{'name': '{vmss-name}', 'properties': {'subnet': {'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/virtualNetworks/{existing-virtual-network-name}/subnets/{existing-subnet-name}'}}}], 'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerNamePrefix': '{vmss-name}'}, 'storageProfile': {'dataDisks': [{'createOption': 'Empty', 'diskSizeGB': 1023, 'lun': 0}, {'createOption': 'Empty', 'diskSizeGB': 1023, 'lun': 1}], 'imageReference': {'offer': 'WindowsServer', 'publisher': 'MicrosoftWindowsServer', 'sku': '2016-Datacenter', 'version': 'latest'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'diskSizeGB': 512, 'managedDisk': {'storageAccountType': 'Standard_LRS'}}}}}, 'sku': {'capacity': 2, 'name': 'Standard_A1_v2', 'tier': 'Standard'}, 'zones': ['1', '3']}).result()
    print(response)
if __name__ == '__main__':
    main()