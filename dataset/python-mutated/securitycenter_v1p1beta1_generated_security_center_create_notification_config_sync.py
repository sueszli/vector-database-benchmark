from google.cloud import securitycenter_v1p1beta1

def sample_create_notification_config():
    if False:
        print('Hello World!')
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.CreateNotificationConfigRequest(parent='parent_value', config_id='config_id_value')
    response = client.create_notification_config(request=request)
    print(response)