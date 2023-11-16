from google.cloud import securitycenter_v1p1beta1

def sample_delete_notification_config():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.DeleteNotificationConfigRequest(name='name_value')
    client.delete_notification_config(request=request)