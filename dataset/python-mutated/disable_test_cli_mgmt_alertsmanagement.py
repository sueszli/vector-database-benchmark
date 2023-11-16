import unittest
import azure.mgmt.alertsmanagement
from devtools_testutils import AzureMgmtTestCase, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtAlertsTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MgmtAlertsTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.alertsmanagement.AlertsManagementClient)

    @unittest.skip('skip')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_alertsmanagement(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        ALERT_ID = 'myAlertId'
        SMART_GROUP_ID = 'mySmartGroupId'
        ACTION_RULE_NAME = 'myActionRule'
        ALERT_RULE_NAME = 'myAlertRule'
        BODY = {'location': 'Global', 'properties': {'scope': {'scope_type': 'ResourceGroup', 'values': ['/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP]}, 'conditions': {'severity': {'operator': 'Equals', 'values': ['Sev0', 'Sev2']}, 'monitor_service': {'operator': 'Equals', 'values': ['Platform', 'Application Insights']}, 'monitor_condition': {'operator': 'Equals', 'values': ['Fired']}, 'target_resource_type': {'operator': 'NotEquals', 'values': ['Microsoft.Compute/VirtualMachines']}}, 'type': 'Suppression', 'suppression_config': {'recurrence_type': 'Daily', 'schedule': {'start_date': '12/09/2018', 'end_date': '12/18/2018', 'start_time': '06:00:00', 'end_time': '14:00:00'}}, 'description': 'Action rule on resource group for daily suppression', 'status': 'Enabled'}}
        result = self.mgmt_client.action_rules.create_update(resource_group_name=RESOURCE_GROUP, action_rule_name=ACTION_RULE_NAME, action_rule=BODY)
        BODY = {'description': 'Sample smart detector alert rule description', 'state': 'Enabled', 'severity': 'Sev3', 'frequency': 'PT5M', 'detector': {'id': 'VMMemoryLeak'}, 'scope': ['/subscriptions/b368ca2f-e298-46b7-b0ab-012281956afa/resourceGroups/MyVms/providers/Microsoft.Compute/virtualMachines/vm1'], 'action_groups': {'custom_email_subject': 'My custom email subject', 'custom_webhook_payload': '{"AlertRuleName":"#alertrulename"}', 'group_ids': ['/subscriptions/b368ca2f-e298-46b7-b0ab-012281956afa/resourcegroups/actionGroups/providers/microsoft.insights/actiongroups/MyActionGroup']}, 'throttling': {'duration': 'PT20M'}}
        result = self.mgmt_client.action_rules.get_by_name(resource_group_name=RESOURCE_GROUP, action_rule_name=ACTION_RULE_NAME)
        result = self.mgmt_client.smart_detector_alert_rules.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.action_rules.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.smart_detector_alert_rules.list()
        result = self.mgmt_client.alerts.get_summary(groupby='severity,alertState')
        result = self.mgmt_client.smart_groups.get_all()
        result = self.mgmt_client.action_rules.list_by_subscription()
        result = self.mgmt_client.alerts.get_all()
        BODY = {'tags': {'new_key': 'newVal'}, 'description': 'New description for patching', 'frequency': 'PT1M'}
        BODY = {'tags': {'key1': 'value1', 'key2': 'value2'}, 'status': 'Disabled'}
        BODY = {'comments': 'Acknowledging smart group'}
        BODY = {'comments': 'Acknowledging alert'}
        result = self.mgmt_client.action_rules.delete(resource_group_name=RESOURCE_GROUP, action_rule_name=ACTION_RULE_NAME)
if __name__ == '__main__':
    unittest.main()