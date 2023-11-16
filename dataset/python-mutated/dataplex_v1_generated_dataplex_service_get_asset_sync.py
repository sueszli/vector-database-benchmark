from google.cloud import dataplex_v1

def sample_get_asset():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.GetAssetRequest(name='name_value')
    response = client.get_asset(request=request)
    print(response)