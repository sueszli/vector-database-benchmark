from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_api_revision_from_existing_api.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', api_id='echo-api;rev=3', parameters={'properties': {'apiRevisionDescription': 'Creating a Revision of an existing API', 'path': 'echo', 'serviceUrl': 'http://echoapi.cloudapp.net/apiv3', 'sourceApiId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.ApiManagement/service/apimService1/apis/echo-api'}}).result()
    print(response)
if __name__ == '__main__':
    main()