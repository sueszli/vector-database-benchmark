from google.analytics import data_v1alpha

def sample_query_audience_list():
    if False:
        return 10
    client = data_v1alpha.AlphaAnalyticsDataClient()
    request = data_v1alpha.QueryAudienceListRequest(name='name_value')
    response = client.query_audience_list(request=request)
    print(response)