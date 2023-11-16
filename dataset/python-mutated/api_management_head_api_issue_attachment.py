from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_head_api_issue_attachment.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_issue_attachment.get_entity_tag(resource_group_name='rg1', service_name='apimService1', api_id='57d2ef278aa04f0888cba3f3', issue_id='57d2ef278aa04f0ad01d6cdc', attachment_id='57d2ef278aa04f0888cba3f3')
    print(response)
if __name__ == '__main__':
    main()