from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python smart_groups_change_state.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='dd91de05-d791-4ceb-b6dc-988682dc7d72')
    response = client.smart_groups.change_state(smart_group_id='a808445e-bb38-4751-85c2-1b109ccc1059', new_state='Acknowledged')
    print(response)
if __name__ == '__main__':
    main()