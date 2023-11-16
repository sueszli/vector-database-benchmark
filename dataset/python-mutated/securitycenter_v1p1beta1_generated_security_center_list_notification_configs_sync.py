from google.cloud import securitycenter_v1p1beta1

def sample_list_notification_configs():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.ListNotificationConfigsRequest(parent='parent_value')
    page_result = client.list_notification_configs(request=request)
    for response in page_result:
        print(response)