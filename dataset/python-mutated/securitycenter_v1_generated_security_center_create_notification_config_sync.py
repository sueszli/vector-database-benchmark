from google.cloud import securitycenter_v1

def sample_create_notification_config():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.CreateNotificationConfigRequest(parent='parent_value', config_id='config_id_value')
    response = client.create_notification_config(request=request)
    print(response)