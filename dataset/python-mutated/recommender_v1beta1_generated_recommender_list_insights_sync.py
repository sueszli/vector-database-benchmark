from google.cloud import recommender_v1beta1

def sample_list_insights():
    if False:
        print('Hello World!')
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.ListInsightsRequest(parent='parent_value')
    page_result = client.list_insights(request=request)
    for response in page_result:
        print(response)