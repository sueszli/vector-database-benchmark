from google.cloud import monitoring_v3

def sample_update_alert_policy():
    if False:
        while True:
            i = 10
    client = monitoring_v3.AlertPolicyServiceClient()
    request = monitoring_v3.UpdateAlertPolicyRequest()
    response = client.update_alert_policy(request=request)
    print(response)