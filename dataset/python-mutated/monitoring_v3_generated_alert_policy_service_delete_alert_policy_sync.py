from google.cloud import monitoring_v3

def sample_delete_alert_policy():
    if False:
        return 10
    client = monitoring_v3.AlertPolicyServiceClient()
    request = monitoring_v3.DeleteAlertPolicyRequest(name='name_value')
    client.delete_alert_policy(request=request)