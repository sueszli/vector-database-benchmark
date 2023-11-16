from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python list_hybrid_runbook_worker.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.hybrid_runbook_workers.list_by_hybrid_runbook_worker_group(resource_group_name='rg', automation_account_name='testaccount', hybrid_runbook_worker_group_name='TestHybridGroup')
    for item in response:
        print(item)
if __name__ == '__main__':
    main()