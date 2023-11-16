from azure.identity import DefaultAzureCredential
from azure.mgmt.alertsmanagement import AlertsManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-alertsmanagement\n# USAGE\n    python create_or_update_prometheus_rule_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AlertsManagementClient(credential=DefaultAzureCredential(), subscription_id='14ddf0c5-77c5-4b53-84f6-e1fa43ad68f7')
    response = client.prometheus_rule_groups.create_or_update(resource_group_name='giladstest', rule_group_name='myPrometheusRuleGroup', parameters={'location': 'East US', 'properties': {'description': 'This is the description of the first rule group', 'rules': [{'expression': 'histogram_quantile(0.99, sum(rate(jobs_duration_seconds_bucket{service="billing-processing"}[5m])) by (job_type))', 'labels': {'team': 'prod'}, 'record': 'job_type:billing_jobs_duration_seconds:99p5m'}, {'actions': [{'actionGroupId': '/subscriptions/14ddf0c5-77c5-4b53-84f6-e1fa43ad68f7/resourcegroups/giladstest/providers/microsoft.insights/notificationgroups/group2', 'actionProperties': {'key11': 'value11', 'key12': 'value12'}}], 'alert': 'Billing_Processing_Very_Slow', 'annotations': {'annotationName1': 'annotationValue1'}, 'expression': 'job_type:billing_jobs_duration_seconds:99p5m > 30', 'for': 'PT5M', 'labels': {'team': 'prod'}, 'resolveConfiguration': {'autoResolved': True, 'timeToResolve': 'PT10M'}, 'severity': 2}], 'scopes': ['/subscriptions/14ddf0c5-77c5-4b53-84f6-e1fa43ad68f7/resourceGroups/giladstest/providers/microsoft.monitor/accounts/myMonitoringAccount']}})
    print(response)
if __name__ == '__main__':
    main()