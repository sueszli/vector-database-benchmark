from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python update_source_control_patch.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.source_control.update(resource_group_name='rg', automation_account_name='sampleAccount9', source_control_name='sampleSourceControl', parameters={'properties': {'autoSync': True, 'branch': 'master', 'description': 'my description', 'folderPath': '/folderOne/folderTwo', 'publishRunbook': True, 'securityToken': {'accessToken': '3a326f7a0dcd343ea58fee21f2fd5fb4c1234567', 'tokenType': 'PersonalAccessToken'}}})
    print(response)
if __name__ == '__main__':
    main()