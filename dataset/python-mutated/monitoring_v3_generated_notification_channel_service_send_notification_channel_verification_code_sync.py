from google.cloud import monitoring_v3

def sample_send_notification_channel_verification_code():
    if False:
        print('Hello World!')
    client = monitoring_v3.NotificationChannelServiceClient()
    request = monitoring_v3.SendNotificationChannelVerificationCodeRequest(name='name_value')
    client.send_notification_channel_verification_code(request=request)