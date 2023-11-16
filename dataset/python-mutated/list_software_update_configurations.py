from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python list_software_update_configurations.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='1a7d4044-286c-4acb-969a-96639265bf2e')
    response = client.software_update_configurations.list(resource_group_name='mygroup', automation_account_name='myaccount')
    print(response)
if __name__ == '__main__':
    main()