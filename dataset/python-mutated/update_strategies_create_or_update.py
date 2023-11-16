from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservicefleet import ContainerServiceFleetMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservicefleet\n# USAGE\n    python update_strategies_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ContainerServiceFleetMgmtClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.fleet_update_strategies.begin_create_or_update(resource_group_name='rg1', fleet_name='fleet1', update_strategy_name='strartegy1', resource={'properties': {'strategy': {'stages': [{'afterStageWaitInSeconds': 3600, 'groups': [{'name': 'group-a'}], 'name': 'stage1'}]}}}).result()
    print(response)
if __name__ == '__main__':
    main()