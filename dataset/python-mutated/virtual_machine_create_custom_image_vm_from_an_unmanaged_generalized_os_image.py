from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_create_custom_image_vm_from_an_unmanaged_generalized_os_image.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_create_or_update(resource_group_name='myResourceGroup', vm_name='{vm-name}', parameters={'location': 'westus', 'properties': {'hardwareProfile': {'vmSize': 'Standard_D1_v2'}, 'networkProfile': {'networkInterfaces': [{'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/networkInterfaces/{existing-nic-name}', 'properties': {'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerName': 'myVM'}, 'storageProfile': {'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'image': {'uri': 'http://{existing-storage-account-name}.blob.core.windows.net/{existing-container-name}/{existing-generalized-os-image-blob-name}.vhd'}, 'name': 'myVMosdisk', 'osType': 'Windows', 'vhd': {'uri': 'http://{existing-storage-account-name}.blob.core.windows.net/{existing-container-name}/myDisk.vhd'}}}}}).result()
    print(response)
if __name__ == '__main__':
    main()