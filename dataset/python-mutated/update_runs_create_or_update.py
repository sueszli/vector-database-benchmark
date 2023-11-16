from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservicefleet import ContainerServiceFleetMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservicefleet\n# USAGE\n    python update_runs_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerServiceFleetMgmtClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.update_runs.begin_create_or_update(resource_group_name='rg1', fleet_name='fleet1', update_run_name='run1', resource={'properties': {'managedClusterUpdate': {'nodeImageSelection': {'type': 'Latest'}, 'upgrade': {'kubernetesVersion': '1.26.1', 'type': 'Full'}}, 'strategy': {'stages': [{'afterStageWaitInSeconds': 3600, 'groups': [{'name': 'group-a'}], 'name': 'stage1'}]}, 'updateStrategyId': '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/rg1/providers/Microsoft.ContainerService/fleets/myFleet/updateStrategies/strategy1'}}).result()
    print(response)
if __name__ == '__main__':
    main()