from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_policy_fragment.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.policy_fragment.begin_create_or_update(resource_group_name='rg1', service_name='apimService1', id='policyFragment1', parameters={'properties': {'description': 'A policy fragment example', 'format': 'xml', 'value': '<fragment><json-to-xml apply="always" consider-accept-header="false" /></fragment>'}}).result()
    print(response)
if __name__ == '__main__':
    main()