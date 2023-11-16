from google.cloud import monitoring_v3

def sample_verify_notification_channel():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.VerifyNotificationChannelRequest(name='name_value', code='code_value')
    response = client.verify_notification_channel(request=request)
    print(response)