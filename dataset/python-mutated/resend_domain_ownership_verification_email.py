from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python resend_domain_ownership_verification_email.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.app_service_certificate_orders.resend_request_emails(resource_group_name='testrg123', certificate_order_name='SampleCertificateOrderName', name_identifier={'name': 'Domain name'})
    print(response)
if __name__ == '__main__':
    main()