from google.analytics import data_v1alpha

def sample_create_audience_list():
    if False:
        return 10
    client = data_v1alpha.AlphaAnalyticsDataClient()
    audience_list = data_v1alpha.AudienceList()
    audience_list.audience = 'audience_value'
    request = data_v1alpha.CreateAudienceListRequest(parent='parent_value', audience_list=audience_list)
    operation = client.create_audience_list(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)