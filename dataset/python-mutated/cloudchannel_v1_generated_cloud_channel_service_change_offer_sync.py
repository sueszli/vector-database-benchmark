from google.cloud import channel_v1

def sample_change_offer():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ChangeOfferRequest(name='name_value', offer='offer_value')
    operation = client.change_offer(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)