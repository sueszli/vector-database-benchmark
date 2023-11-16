from azure.identity import DefaultAzureCredential
from azure.mgmt.appplatform import AppPlatformManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appplatform\n# USAGE\n    python gateways_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AppPlatformManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.gateways.begin_create_or_update(resource_group_name='myResourceGroup', service_name='myservice', gateway_name='default', gateway_resource={'properties': {'public': True, 'resourceRequests': {'cpu': '1', 'memory': '1G'}}, 'sku': {'capacity': 2, 'name': 'E0', 'tier': 'Enterprise'}}).result()
    print(response)
if __name__ == '__main__':
    main()