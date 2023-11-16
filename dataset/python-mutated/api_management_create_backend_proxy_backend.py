from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-apimanagement\n# USAGE\n    python api_management_create_backend_proxy_backend.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ApiManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.backend.create_or_update(resource_group_name='rg1', service_name='apimService1', backend_id='proxybackend', parameters={'properties': {'credentials': {'authorization': {'parameter': 'opensesma', 'scheme': 'Basic'}, 'header': {'x-my-1': ['val1', 'val2']}, 'query': {'sv': ['xx', 'bb', 'cc']}}, 'description': 'description5308', 'protocol': 'http', 'proxy': {'password': '<password>', 'url': 'http://192.168.1.1:8080', 'username': 'Contoso\\admin'}, 'tls': {'validateCertificateChain': True, 'validateCertificateName': True}, 'url': 'https://backendname2644/'}})
    print(response)
if __name__ == '__main__':
    main()