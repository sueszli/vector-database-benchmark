from azure.identity import DefaultAzureCredential
from azure.mgmt.billing import BillingManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billing\n# USAGE\n    python billing_profile_invoice_download.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = BillingManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.invoices.begin_download_multiple_billing_profile_invoices(billing_account_name='{billingAccountName}', download_urls=['https://management.azure.com/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/invoices/{invoiceName}/download?downloadToken={downloadToken}&useCache=True&api-version=2020-05-01', 'https://management.azure.com/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/invoices/{invoiceName}/download?downloadToken={downloadToken}&useCache=True&api-version=2020-05-01', 'https://management.azure.com/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/invoices/{invoiceName}/download?downloadToken={downloadToken}&useCache=True&api-version=2020-05-01']).result()
    print(response)
if __name__ == '__main__':
    main()