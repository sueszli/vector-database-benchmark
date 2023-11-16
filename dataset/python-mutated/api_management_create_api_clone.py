from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_api_clone.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', api_id='echo-api2', parameters={'properties': {'description': 'Copy of Existing Echo Api including Operations.', 'displayName': 'Echo API2', 'isCurrent': True, 'path': 'echo2', 'protocols': ['http', 'https'], 'serviceUrl': 'http://echoapi.cloudapp.net/api', 'sourceApiId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.ApiManagement/service/apimService1/apis/58a4aeac497000007d040001', 'subscriptionRequired': True}}).result()
    print(response)
if __name__ == '__main__':
    main()