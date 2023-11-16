from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_create_with_network_interface_configuration_dns_settings.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_create_or_update(resource_group_name='myResourceGroup', vm_name='myVM', parameters={'location': 'westus', 'properties': {'hardwareProfile': {'vmSize': 'Standard_D1_v2'}, 'networkProfile': {'networkApiVersion': '2020-11-01', 'networkInterfaceConfigurations': [{'name': '{nic-config-name}', 'properties': {'deleteOption': 'Delete', 'ipConfigurations': [{'name': '{ip-config-name}', 'properties': {'primary': True, 'publicIPAddressConfiguration': {'name': '{publicIP-config-name}', 'properties': {'deleteOption': 'Detach', 'dnsSettings': {'domainNameLabel': 'aaaaa', 'domainNameLabelScope': 'TenantReuse'}, 'publicIPAllocationMethod': 'Static'}, 'sku': {'name': 'Basic', 'tier': 'Global'}}}}], 'primary': True}}]}, 'osProfile': {'adminPassword': '{your-password}', 'adminUsername': '{your-username}', 'computerName': 'myVM'}, 'storageProfile': {'imageReference': {'offer': 'WindowsServer', 'publisher': 'MicrosoftWindowsServer', 'sku': '2016-Datacenter', 'version': 'latest'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'managedDisk': {'storageAccountType': 'Standard_LRS'}, 'name': 'myVMosdisk'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()