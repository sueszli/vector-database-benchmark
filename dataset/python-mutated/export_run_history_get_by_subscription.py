from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python export_run_history_get_by_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.exports.get_execution_history(scope='subscriptions/00000000-0000-0000-0000-000000000000', export_name='TestExport')
    print(response)
if __name__ == '__main__':
    main()