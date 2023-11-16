from google.analytics import data_v1alpha

def sample_list_audience_lists():
    if False:
        for i in range(10):
            print('nop')
    client = data_v1alpha.AlphaAnalyticsDataClient()
    request = data_v1alpha.ListAudienceListsRequest(parent='parent_value')
    page_result = client.list_audience_lists(request=request)
    for response in page_result:
        print(response)