from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_api_with_multiple_auth_servers.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', api_id='tempgroup', parameters={'properties': {'authenticationSettings': {'oAuth2AuthenticationSettings': [{'authorizationServerId': 'authorizationServerId2283', 'scope': 'oauth2scope2580'}, {'authorizationServerId': 'authorizationServerId2284', 'scope': 'oauth2scope2581'}]}, 'description': 'apidescription5200', 'displayName': 'apiname1463', 'path': 'newapiPath', 'protocols': ['https', 'http'], 'serviceUrl': 'http://newechoapi.cloudapp.net/api', 'subscriptionKeyParameterNames': {'header': 'header4520', 'query': 'query3037'}}}).result()
    print(response)
if __name__ == '__main__':
    main()