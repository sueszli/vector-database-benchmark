from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_update_service_to_new_vnet_and_azs.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_management_service.begin_update(resource_group_name='rg1', service_name='apimService1', parameters={'properties': {'additionalLocations': [{'location': 'Australia East', 'publicIpAddressId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.Network/publicIPAddresses/apim-australia-east-publicip', 'sku': {'capacity': 3, 'name': 'Premium'}, 'virtualNetworkConfiguration': {'subnetResourceId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.Network/virtualNetworks/apimaeavnet/subnets/default'}, 'zones': ['1', '2', '3']}], 'publicIpAddressId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.Network/publicIPAddresses/publicip-apim-japan-east', 'virtualNetworkConfiguration': {'subnetResourceId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.Network/virtualNetworks/vnet-apim-japaneast/subnets/apim2'}, 'virtualNetworkType': 'External'}, 'sku': {'capacity': 3, 'name': 'Premium'}, 'zones': ['1', '2', '3']}).result()
    print(response)
if __name__ == '__main__':
    main()