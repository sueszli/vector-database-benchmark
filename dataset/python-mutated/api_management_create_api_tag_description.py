from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_api_tag_description.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_tag_description.create_or_update(resource_group_name='rg1', service_name='apimService1', api_id='5931a75ae4bbd512a88c680b', tag_description_id='tagId1', parameters={'properties': {'description': "Some description that will be displayed for operation's tag if the tag is assigned to operation of the API", 'externalDocsDescription': 'Description of the external docs resource', 'externalDocsUrl': 'http://some.url/additionaldoc'}})
    print(response)
if __name__ == '__main__':
    main()