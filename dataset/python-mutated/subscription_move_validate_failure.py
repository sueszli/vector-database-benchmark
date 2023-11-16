from azure.identity import DefaultAzureCredential
from azure.mgmt.billing import BillingManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billing\n# USAGE\n    python subscription_move_validate_failure.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = BillingManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscriptionId}')
    response = client.billing_subscriptions.validate_move(billing_account_name='{billingAccountName}', parameters={'destinationInvoiceSectionId': '/providers/Microsoft.Billing/billingAccounts/{billingAccountName}/billingProfiles/{billingProfileName}/invoiceSections/{newInvoiceSectionName}'})
    print(response)
if __name__ == '__main__':
    main()