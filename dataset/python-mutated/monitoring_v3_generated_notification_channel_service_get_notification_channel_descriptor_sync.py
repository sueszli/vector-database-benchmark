from google.cloud import monitoring_v3

def sample_get_notification_channel_descriptor():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.GetNotificationChannelDescriptorRequest(name='name_value')
    response = client.get_notification_channel_descriptor(request=request)
    print(response)