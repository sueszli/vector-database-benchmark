from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python diagnostics_get_app_service_certificate_order_detector_response.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='5700fc96-77b4-4f8d-afce-c353d8c443bd')
    response = client.certificate_orders_diagnostics.get_app_service_certificate_order_detector_response(resource_group_name='Sample-WestUSResourceGroup', certificate_order_name='SampleCertificateOrderName', detector_name='AutoRenewStatus')
    print(response)
if __name__ == '__main__':
    main()