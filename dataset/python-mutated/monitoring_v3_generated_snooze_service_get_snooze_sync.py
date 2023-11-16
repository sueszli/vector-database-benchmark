from google.cloud import monitoring_v3

def sample_get_snooze():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.SnoozeServiceClient()
    request = monitoring_v3.GetSnoozeRequest(name='name_value')
    response = client.get_snooze(request=request)
    print(response)