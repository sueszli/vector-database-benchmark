from azure.identity import DefaultAzureCredential
from azure.mgmt.billing import BillingManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-billing\n# USAGE\n    python put_instruction.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = BillingManagementClient(credential=DefaultAzureCredential(), subscription_id='SUBSCRIPTION_ID')
    response = client.instructions.put(billing_account_name='{billingAccountName}', billing_profile_name='{billingProfileName}', instruction_name='{instructionName}', parameters={'properties': {'amount': 5000, 'endDate': '2020-12-30T21:26:47.997Z', 'startDate': '2019-12-30T21:26:47.997Z'}})
    print(response)
if __name__ == '__main__':
    main()