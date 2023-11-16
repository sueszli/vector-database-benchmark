from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_api_with_open_id_connect.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', api_id='tempgroup', parameters={'properties': {'authenticationSettings': {'openid': {'bearerTokenSendingMethods': ['authorizationHeader'], 'openidProviderId': 'testopenid'}}, 'description': 'This is a sample server Petstore server.  You can find out more about Swagger at `http://swagger.io <http://swagger.io>`_ or on `irc.freenode.net, #swagger <http://swagger.io/irc/>`_.  For this sample, you can use the api key ``special-key`` to test the authorization filters.', 'displayName': 'Swagger Petstore', 'path': 'petstore', 'protocols': ['https'], 'serviceUrl': 'http://petstore.swagger.io/v2', 'subscriptionKeyParameterNames': {'header': 'Ocp-Apim-Subscription-Key', 'query': 'subscription-key'}}}).result()
    print(response)
if __name__ == '__main__':
    main()