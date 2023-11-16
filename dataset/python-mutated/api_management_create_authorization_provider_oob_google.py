from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_authorization_provider_oob_google.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.authorization_provider.create_or_update(resource_group_name='rg1', service_name='apimService1', authorization_provider_id='google', parameters={'properties': {'displayName': 'google', 'identityProvider': 'google', 'oauth2': {'grantTypes': {'authorizationCode': {'clientId': '508791967882-5qv6o2i99a75un7329vlegtk78kr766h.apps.googleusercontent.com', 'clientSecret': 'qDN0VyVFjU1OsOyT5Kz8ce', 'scopes': 'openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email'}}, 'redirectUrl': 'https://authorization-manager.consent.azure-apim.net/redirect/apim/apimService1'}}})
    print(response)
if __name__ == '__main__':
    main()