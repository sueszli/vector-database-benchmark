from google.cloud import monitoring_v3

def sample_create_alert_policy():
    if False:
        return 10
    client = monitoring_v3.AlertPolicyServiceClient()
    request = monitoring_v3.CreateAlertPolicyRequest(name='name_value')
    response = client.create_alert_policy(request=request)
    print(response)