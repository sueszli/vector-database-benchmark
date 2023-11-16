from google.cloud import monitoring_v3

def sample_list_notification_channel_descriptors():
    if False:
        print('Hello World!')
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.ListNotificationChannelDescriptorsRequest(name='name_value')
    page_result = client.list_notification_channel_descriptors(request=request)
    for response in page_result:
        print(response)