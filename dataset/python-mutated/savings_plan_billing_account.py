from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python savings_plan_billing_account.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.benefit_utilization_summaries.list_by_billing_account_id(billing_account_id='12345')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()