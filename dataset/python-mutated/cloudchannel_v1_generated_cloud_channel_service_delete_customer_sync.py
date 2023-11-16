from google.cloud import channel_v1

def sample_delete_customer():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.DeleteCustomerRequest(name='name_value')
    client.delete_customer(request=request)