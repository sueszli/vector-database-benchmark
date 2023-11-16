from azure.identity import DefaultAzureCredential
from azure.mgmt.azurestack import AzureStackManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-azurestack\n# USAGE\n    python get_post.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AzureStackManagementClient(credential=DefaultAzureCredential(), subscription_id='dd8597b4-8739-4467-8b10-f8679f62bfbf')
    response = client.products.get_product(resource_group='azurestack', registration_name='testregistration', product_name='Microsoft.OSTCExtensions.VMAccessForLinux.1.4.7.1')
    print(response)
if __name__ == '__main__':
    main()