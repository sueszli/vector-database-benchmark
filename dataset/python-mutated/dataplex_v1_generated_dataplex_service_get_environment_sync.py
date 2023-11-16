from google.cloud import dataplex_v1

def sample_get_environment():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)