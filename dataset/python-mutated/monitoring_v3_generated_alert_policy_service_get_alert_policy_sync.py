from google.cloud import monitoring_v3

def sample_get_alert_policy():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.AlertPolicyServiceClient()
    request = monitoring_v3.GetAlertPolicyRequest(name='name_value')
    response = client.get_alert_policy(request=request)
    print(response)