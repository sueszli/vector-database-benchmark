from azure.identity import DefaultAzureCredential
from azure.mgmt.billing import BillingManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billing\n# USAGE\n    python billing_profile_with_expand.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = BillingManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.billing_profiles.get(billing_account_name='{billingAccountName}', billing_profile_name='{billingProfileName}')
    print(response)
if __name__ == '__main__':
    main()