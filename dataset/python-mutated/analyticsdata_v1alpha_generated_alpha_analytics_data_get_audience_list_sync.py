from google.analytics import data_v1alpha

def sample_get_audience_list():
    if False:
        for i in range(10):
            print('nop')
    client = data_v1alpha.AlphaAnalyticsDataClient()
    request = data_v1alpha.GetAudienceListRequest(name='name_value')
    response = client.get_audience_list(request=request)
    print(response)