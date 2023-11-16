from google.analytics import data_v1beta

def sample_get_metadata():
    if False:
        return 10
    client = data_v1beta.BetaAnalyticsDataClient()
    request = data_v1beta.GetMetadataRequest(name='name_value')
    response = client.get_metadata(request=request)
    print(response)