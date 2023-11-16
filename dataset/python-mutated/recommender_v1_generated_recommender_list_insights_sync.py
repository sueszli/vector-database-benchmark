from google.cloud import recommender_v1

def sample_list_insights():
    if False:
        i = 10
        return i + 15
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.ListInsightsRequest(parent='parent_value')
    page_result = client.list_insights(request=request)
    for response in page_result:
        print(response)