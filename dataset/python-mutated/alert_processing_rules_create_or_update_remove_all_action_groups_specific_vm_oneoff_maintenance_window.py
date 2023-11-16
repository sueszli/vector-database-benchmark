from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python alert_processing_rules_create_or_update_remove_all_action_groups_specific_vm_oneoff_maintenance_window.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='subId1')
    response = client.alert_processing_rules.create_or_update(resource_group_name='alertscorrelationrg', alert_processing_rule_name='RemoveActionGroupsMaintenanceWindow', alert_processing_rule={'location': 'Global', 'properties': {'actions': [{'actionType': 'RemoveAllActionGroups'}], 'description': 'Removes all ActionGroups from all Alerts on VMName during the maintenance window', 'enabled': True, 'schedule': {'effectiveFrom': '2021-04-15T18:00:00', 'effectiveUntil': '2021-04-15T20:00:00', 'timeZone': 'Pacific Standard Time'}, 'scopes': ['/subscriptions/subId1/resourceGroups/RGId1/providers/Microsoft.Compute/virtualMachines/VMName']}, 'tags': {}})
    print(response)
if __name__ == '__main__':
    main()