from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python deserialize_graph_runbook_content.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.convert_graph_runbook_content(resource_group_name='rg', automation_account_name='MyAutomationAccount', parameters={'rawContent': {'runbookDefinition': 'AAEAAADAQAAAAAAAAAMAgAAAGJPcmNoZXN0cmF0b3IuR3JhcGhSdW5ib29rLk1vZGVsLCBWZXJzaW9uPTcuMy4wLjAsIEN1bHR....', 'runbookType': 'GraphPowerShell', 'schemaVersion': '1.10'}})
    print(response)
if __name__ == '__main__':
    main()