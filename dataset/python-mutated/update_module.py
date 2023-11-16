from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python update_module.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.module.update(resource_group_name='rg', automation_account_name='MyAutomationAccount', module_name='MyModule', parameters={'properties': {'contentLink': {'contentHash': {'algorithm': 'sha265', 'value': '07E108A962B81DD9C9BAA89BB47C0F6EE52B29E83758B07795E408D258B2B87A'}, 'uri': 'https://teststorage.blob.core.windows.net/mycontainer/MyModule.zip', 'version': '1.0.0.0'}}})
    print(response)
if __name__ == '__main__':
    main()