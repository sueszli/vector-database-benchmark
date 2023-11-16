from google.cloud import monitoring_v3

def sample_get_notification_channel():
    if False:
        return 10
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.GetNotificationChannelRequest(name='name_value')
    response = client.get_notification_channel(request=request)
    print(response)