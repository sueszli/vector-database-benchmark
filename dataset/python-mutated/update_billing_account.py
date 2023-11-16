from azure.identity import DefaultAzureCredential
from azure.mgmt.billing import BillingManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billing\n# USAGE\n    python update_billing_account.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = BillingManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.billing_accounts.begin_update(billing_account_name='{billingAccountName}', parameters={'properties': {'displayName': 'Test Account', 'soldTo': {'addressLine1': 'Test Address 1', 'city': 'Redmond', 'companyName': 'Contoso', 'country': 'US', 'firstName': 'Test', 'lastName': 'User', 'postalCode': '12345', 'region': 'WA'}}}).result()
    print(response)
if __name__ == '__main__':
    main()