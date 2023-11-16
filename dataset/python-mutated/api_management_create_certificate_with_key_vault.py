from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_certificate_with_key_vault.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.certificate.create_or_update(resource_group_name='rg1', service_name='apimService1', certificate_id='templateCertkv', parameters={'properties': {'keyVault': {'identityClientId': 'ceaa6b06-c00f-43ef-99ac-f53d1fe876a0', 'secretIdentifier': 'https://rpbvtkeyvaultintegration.vault-int.azure-int.net/secrets/msitestingCert'}}})
    print(response)
if __name__ == '__main__':
    main()