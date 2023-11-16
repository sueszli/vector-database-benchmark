from google.cloud import monitoring_v3

def sample_delete_notification_channel():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.DeleteNotificationChannelRequest(name='name_value')
    client.delete_notification_channel(request=request)