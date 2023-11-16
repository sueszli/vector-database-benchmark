from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_get_api_tag_description.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_tag_description.get(resource_group_name='rg1', service_name='apimService1', api_id='59d6bb8f1f7fab13dc67ec9b', tag_description_id='59306a29e4bbd510dc24e5f9')
    print(response)
if __name__ == '__main__':
    main()