from google.cloud import securitycenter_v1

def sample_get_notification_config():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetNotificationConfigRequest(name='name_value')
    response = client.get_notification_config(request=request)
    print(response)