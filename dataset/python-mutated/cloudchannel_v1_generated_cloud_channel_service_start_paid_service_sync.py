from google.cloud import channel_v1

def sample_start_paid_service():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.StartPaidServiceRequest(name='name_value')
    operation = client.start_paid_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)