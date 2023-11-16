from google.cloud import monitoring_v3

def sample_create_notification_channel():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.CreateNotificationChannelRequest(name='name_value')
    response = client.create_notification_channel(request=request)
    print(response)