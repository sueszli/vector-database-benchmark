from google.cloud import dataplex_v1

def sample_get_zone():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.GetZoneRequest(name='name_value')
    response = client.get_zone(request=request)
    print(response)