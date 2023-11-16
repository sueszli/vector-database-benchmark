from google.cloud import securitycenter_v1p1beta1

def sample_update_notification_config():
    if False:
        while True:
            i = 10
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.UpdateNotificationConfigRequest()
    response = client.update_notification_config(request=request)
    print(response)