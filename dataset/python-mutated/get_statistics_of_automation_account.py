from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python get_statistics_of_automation_account.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.statistics.list_by_automation_account(resource_group_name='rg', automation_account_name='myAutomationAccount11')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()