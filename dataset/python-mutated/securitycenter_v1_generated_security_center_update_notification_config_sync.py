from google.cloud import securitycenter_v1

def sample_update_notification_config():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.UpdateNotificationConfigRequest()
    response = client.update_notification_config(request=request)
    print(response)