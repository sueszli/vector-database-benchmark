from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_create_with_scheduled_events_profile.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_create_or_update(resource_group_name='myResourceGroup', vm_name='myVM', parameters={'location': 'westus', 'properties': {'diagnosticsProfile': {'bootDiagnostics': {'enabled': True, 'storageUri': 'http://{existing-storage-account-name}.blob.core.windows.net'}}, 'hardwareProfile': {'vmSize': 'Standard_D1_v2'}, 'networkProfile': {'networkInterfaces': [{'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/networkInterfaces/{existing-nic-name}', 'properties': {'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerName': 'myVM'}, 'scheduledEventsProfile': {'osImageNotificationProfile': {'enable': True, 'notBeforeTimeout': 'PT15M'}, 'terminateNotificationProfile': {'enable': True, 'notBeforeTimeout': 'PT10M'}}, 'storageProfile': {'imageReference': {'offer': 'WindowsServer', 'publisher': 'MicrosoftWindowsServer', 'sku': '2016-Datacenter', 'version': 'latest'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'managedDisk': {'storageAccountType': 'Standard_LRS'}, 'name': 'myVMosdisk'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()