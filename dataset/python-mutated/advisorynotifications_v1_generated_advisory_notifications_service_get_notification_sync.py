from google.cloud import advisorynotifications_v1

def sample_get_notification():
    if False:
        print('Hello World!')
    client = advisorynotifications_v1.AdvisoryNotificationsServiceClient()
    request = advisorynotifications_v1.GetNotificationRequest(name='name_value')
    response = client.get_notification(request=request)
    print(response)