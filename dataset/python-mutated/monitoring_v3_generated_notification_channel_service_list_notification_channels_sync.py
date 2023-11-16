from google.cloud import monitoring_v3

def sample_list_notification_channels():
    if False:
        while True:
            i = 10
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.ListNotificationChannelsRequest(name='name_value')
    page_result = client.list_notification_channels(request=request)
    for response in page_result:
        print(response)