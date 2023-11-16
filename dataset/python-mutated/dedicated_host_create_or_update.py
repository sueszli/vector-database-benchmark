from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python dedicated_host_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.dedicated_hosts.begin_create_or_update(resource_group_name='myResourceGroup', host_group_name='myDedicatedHostGroup', host_name='myDedicatedHost', parameters={'location': 'westus', 'properties': {'platformFaultDomain': 1}, 'sku': {'name': 'DSv3-Type1'}, 'tags': {'department': 'HR'}}).result()
    print(response)
if __name__ == '__main__':
    main()