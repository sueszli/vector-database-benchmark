from azure.identity import DefaultAzureCredential
from azure.mgmt.dashboard import DashboardManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-dashboard\n# USAGE\n    python grafana_get.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = DashboardManagementClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.grafana.get(resource_group_name='myResourceGroup', workspace_name='myWorkspace')
    print(response)
if __name__ == '__main__':
    main()