from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_api_issue_comment.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_issue_comment.create_or_update(resource_group_name='rg1', service_name='apimService1', api_id='57d1f7558aa04f15146d9d8a', issue_id='57d2ef278aa04f0ad01d6cdc', comment_id='599e29ab193c3c0bd0b3e2fb', parameters={'properties': {'createdDate': '2018-02-01T22:21:20.467Z', 'text': 'Issue comment.', 'userId': '/subscriptions/subid/resourceGroups/rg1/providers/Microsoft.ApiManagement/service/apimService1/users/1'}})
    print(response)
if __name__ == '__main__':
    main()