from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_authorization_aad_auth_code.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.authorization.create_or_update(resource_group_name='rg1', service_name='apimService1', authorization_provider_id='aadwithauthcode', authorization_id='authz2', parameters={'properties': {'authorizationType': 'OAuth2', 'oauth2grantType': 'AuthorizationCode'}})
    print(response)
if __name__ == '__main__':
    main()