from google.cloud import monitoring_v3

def sample_update_notification_channel():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.UpdateNotificationChannelRequest()
    response = client.update_notification_channel(request=request)
    print(response)