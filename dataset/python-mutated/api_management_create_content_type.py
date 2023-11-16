from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_content_type.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.content_type.create_or_update(resource_group_name='rg1', service_name='apimService1', content_type_id='page', parameters={'properties': {'description': 'A regular page', 'name': 'Page', 'schema': {'additionalProperties': False, 'properties': {'en_us': {'additionalProperties': False, 'properties': {'description': {'description': 'Page description. This property gets included in SEO attributes.', 'indexed': True, 'title': 'Description', 'type': 'string'}, 'documentId': {'description': 'Reference to page content document.', 'title': 'Document ID', 'type': 'string'}, 'keywords': {'description': 'Page keywords. This property gets included in SEO attributes.', 'indexed': True, 'title': 'Keywords', 'type': 'string'}, 'permalink': {'description': "Page permalink, e.g. '/about'.", 'indexed': True, 'title': 'Permalink', 'type': 'string'}, 'title': {'description': 'Page title. This property gets included in SEO attributes.', 'indexed': True, 'title': 'Title', 'type': 'string'}}, 'required': ['title', 'permalink', 'documentId'], 'type': 'object'}}}, 'version': '1.0.0'}})
    print(response)
if __name__ == '__main__':
    main()