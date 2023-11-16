from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservicefleet import ContainerServiceFleetMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservicefleet\n# USAGE\n    python update_runs_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerServiceFleetMgmtClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    client.update_runs.begin_delete(resource_group_name='rg1', fleet_name='fleet1', update_run_name='run1').result()
if __name__ == '__main__':
    main()