from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_update_api_operation.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.api_operation.update(resource_group_name='rg1', service_name='apimService1', api_id='echo-api', operation_id='operationId', if_match='*', parameters={'properties': {'displayName': 'Retrieve resource', 'method': 'GET', 'request': {'queryParameters': [{'defaultValue': 'sample', 'description': 'A sample parameter that is required and has a default value of "sample".', 'name': 'param1', 'required': True, 'type': 'string', 'values': ['sample']}]}, 'responses': [{'description': 'Returned in all cases.', 'headers': [], 'representations': [], 'statusCode': 200}, {'description': 'Server Error.', 'headers': [], 'representations': [], 'statusCode': 500}], 'templateParameters': [], 'urlTemplate': '/resource'}})
    print(response)
if __name__ == '__main__':
    main()