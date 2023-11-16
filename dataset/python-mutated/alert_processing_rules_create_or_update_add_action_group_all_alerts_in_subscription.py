from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python alert_processing_rules_create_or_update_add_action_group_all_alerts_in_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='subId1')
    response = client.alert_processing_rules.create_or_update(resource_group_name='alertscorrelationrg', alert_processing_rule_name='AddActionGroupToSubscription', alert_processing_rule={'location': 'Global', 'properties': {'actions': [{'actionGroupIds': ['/subscriptions/subId1/resourcegroups/RGId1/providers/microsoft.insights/actiongroups/ActionGroup1'], 'actionType': 'AddActionGroups'}], 'description': 'Add ActionGroup1 to all alerts in the subscription', 'enabled': True, 'scopes': ['/subscriptions/subId1']}, 'tags': {}})
    print(response)
if __name__ == '__main__':
    main()