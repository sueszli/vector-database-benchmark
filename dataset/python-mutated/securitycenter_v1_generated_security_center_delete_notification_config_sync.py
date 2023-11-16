from google.cloud import securitycenter_v1

def sample_delete_notification_config():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.DeleteNotificationConfigRequest(name='name_value')
    client.delete_notification_config(request=request)