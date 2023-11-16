from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python reissue_app_service_certificate_order.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.app_service_certificate_orders.reissue(resource_group_name='testrg123', certificate_order_name='SampleCertificateOrderName', reissue_certificate_order_request={'properties': {'csr': 'CSR1223238Value', 'delayExistingRevokeInHours': 2, 'isPrivateKeyExternal': False, 'keySize': 2048}})
    print(response)
if __name__ == '__main__':
    main()