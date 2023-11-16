from google.cloud import monitoring_v3

def sample_get_notification_channel_verification_code():
    if False:
        while True:
            i = 10
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.GetNotificationChannelVerificationCodeRequest(name='name_value')
    response = client.get_notification_channel_verification_code(request=request)
    print(response)