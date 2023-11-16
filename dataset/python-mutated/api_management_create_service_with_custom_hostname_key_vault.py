from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_service_with_custom_hostname_key_vault.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_management_service.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', parameters={'identity': {'type': 'UserAssigned', 'userAssignedIdentities': {'/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id1': {}}}, 'location': 'North Europe', 'properties': {'apiVersionConstraint': {'minApiVersion': '2019-01-01'}, 'hostnameConfigurations': [{'defaultSslBinding': True, 'hostName': 'gateway1.msitesting.net', 'identityClientId': '329419bc-adec-4dce-9568-25a6d486e468', 'keyVaultId': 'https://rpbvtkeyvaultintegration.vault.azure.net/secrets/msitestingCert', 'type': 'Proxy'}, {'hostName': 'mgmt.msitesting.net', 'identityClientId': '329419bc-adec-4dce-9568-25a6d486e468', 'keyVaultId': 'https://rpbvtkeyvaultintegration.vault.azure.net/secrets/msitestingCert', 'type': 'Management'}, {'hostName': 'portal1.msitesting.net', 'identityClientId': '329419bc-adec-4dce-9568-25a6d486e468', 'keyVaultId': 'https://rpbvtkeyvaultintegration.vault.azure.net/secrets/msitestingCert', 'type': 'Portal'}], 'publisherEmail': 'apim@autorestsdk.com', 'publisherName': 'autorestsdk', 'virtualNetworkType': 'None'}, 'sku': {'capacity': 1, 'name': 'Premium'}, 'tags': {'tag1': 'value1', 'tag2': 'value2', 'tag3': 'value3'}}).result()
    print(response)
if __name__ == '__main__':
    main()