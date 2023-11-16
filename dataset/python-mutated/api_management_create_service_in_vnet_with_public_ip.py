from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_service_in_vnet_with_public_ip.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_management_service.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', parameters={'location': 'East US 2 EUAP', 'properties': {'publicIpAddressId': '/subscriptions/subid/resourceGroups/rgName/providers/Microsoft.Network/publicIPAddresses/apimazvnet', 'publisherEmail': 'apim@autorestsdk.com', 'publisherName': 'autorestsdk', 'virtualNetworkConfiguration': {'subnetResourceId': '/subscriptions/subid/resourceGroups/rgName/providers/Microsoft.Network/virtualNetworks/apimcus/subnets/tenant'}, 'virtualNetworkType': 'External'}, 'sku': {'capacity': 2, 'name': 'Premium'}, 'tags': {'tag1': 'value1', 'tag2': 'value2', 'tag3': 'value3'}, 'zones': ['1', '2']}).result()
    print(response)
if __name__ == '__main__':
    main()