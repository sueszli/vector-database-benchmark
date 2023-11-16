from google.cloud import securitycenter_v1

def sample_list_notification_configs():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.ListNotificationConfigsRequest(parent='parent_value')
    page_result = client.list_notification_configs(request=request)
    for response in page_result:
        print(response)