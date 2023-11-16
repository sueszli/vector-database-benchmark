from google.cloud import securitycenter_v1p1beta1

def sample_get_notification_config():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.GetNotificationConfigRequest(name='name_value')
    response = client.get_notification_config(request=request)
    print(response)