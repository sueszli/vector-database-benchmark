from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_named_value_with_key_vault.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.named_value.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', named_value_id='testprop6', parameters={'properties': {'displayName': 'prop6namekv', 'keyVault': {'identityClientId': 'ceaa6b06-c00f-43ef-99ac-f53d1fe876a0', 'secretIdentifier': 'https://contoso.vault.azure.net/secrets/aadSecret'}, 'secret': True, 'tags': ['foo', 'bar']}}).result()
    print(response)
if __name__ == '__main__':
    main()