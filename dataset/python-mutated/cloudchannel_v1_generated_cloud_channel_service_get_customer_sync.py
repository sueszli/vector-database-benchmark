from google.cloud import channel_v1

def sample_get_customer():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.GetCustomerRequest(name='name_value')
    response = client.get_customer(request=request)
    print(response)