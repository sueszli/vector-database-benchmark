from google.cloud import eventarc_v1

def sample_get_provider():
    if False:
        return 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.GetProviderRequest(name='name_value')
    response = client.get_provider(request=request)
    print(response)