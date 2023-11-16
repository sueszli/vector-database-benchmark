from google.cloud import monitoring_v3

def sample_list_alert_policies():
    if False:
        while True:
            i = 10
    client = monitoring_v3.AlertPolicyServiceClient()
    request = monitoring_v3.ListAlertPoliciesRequest(name='name_value')
    page_result = client.list_alert_policies(request=request)
    for response in page_result:
        print(response)