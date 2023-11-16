from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_create_with_ssh_authentication.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machines.begin_create_or_update(resource_group_name='myResourceGroup', vm_name='myVM', parameters={'location': 'westus', 'properties': {'hardwareProfile': {'vmSize': 'Standard_D1_v2'}, 'networkProfile': {'networkInterfaces': [{'id': '/subscriptions/{subscription-id}/resourceGroups/myResourceGroup/providers/Microsoft.Network/networkInterfaces/{existing-nic-name}', 'properties': {'primary': True}}]}, 'osProfile': {'adminUsername': '{your-username}', 'computerName': 'myVM', 'linuxConfiguration': {'disablePasswordAuthentication': True, 'ssh': {'publicKeys': [{'keyData': 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCeClRAk2ipUs/l5voIsDC5q9RI+YSRd1Bvd/O+axgY4WiBzG+4FwJWZm/mLLe5DoOdHQwmU2FrKXZSW4w2sYE70KeWnrFViCOX5MTVvJgPE8ClugNl8RWth/tU849DvM9sT7vFgfVSHcAS2yDRyDlueii+8nF2ym8XWAPltFVCyLHRsyBp5YPqK8JFYIa1eybKsY3hEAxRCA+/7bq8et+Gj3coOsuRmrehav7rE6N12Pb80I6ofa6SM5XNYq4Xk0iYNx7R3kdz0Jj9XgZYWjAHjJmT0gTRoOnt6upOuxK7xI/ykWrllgpXrCPu3Ymz+c+ujaqcxDopnAl2lmf69/J1', 'path': '/home/{your-username}/.ssh/authorized_keys'}]}}}, 'storageProfile': {'imageReference': {'offer': '{image_offer}', 'publisher': '{image_publisher}', 'sku': '{image_sku}', 'version': 'latest'}, 'osDisk': {'caching': 'ReadWrite', 'createOption': 'FromImage', 'managedDisk': {'storageAccountType': 'Standard_LRS'}, 'name': 'myVMosdisk'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()