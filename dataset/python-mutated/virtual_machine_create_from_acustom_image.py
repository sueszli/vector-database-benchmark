from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_create_from_acustom_image.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_create_or_update(resource_group_name='myResourceGroup', vm_name='myVM', parameters={'location': 'westus', 'properties': {'hardwareProfile': {'vmSize': 'Standard_D1_v2'}, 'networkProfile': {'networkInterfaces': [{'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/networkInterfaces/{existing-nic-name}', 'properties': {'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerName': 'myVM'}, 'storageProfile': {'imageReference': {'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Compute/images/{existing-custom-image-name}'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'managedDisk': {'storageAccountType': 'Standard_LRS'}, 'name': 'myVMosdisk'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()