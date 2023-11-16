from azure.identity import DefaultAzureCredential
from azure.mgmt.automation import AutomationClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automation\n# USAGE\n    python get_software_update_configuration_machine_run_by_id.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AutomationClient(credential=DefaultAzureCredential(), subscription_id='51766542-3ed7-4a72-a187-0c8ab644ddab')
    response = client.software_update_configuration_machine_runs.get_by_id(resource_group_name='mygroup', automation_account_name='myaccount', software_update_configuration_machine_run_id='ca440719-34a4-4234-a1a9-3f84faf7788f')
    print(response)
if __name__ == '__main__':
    main()