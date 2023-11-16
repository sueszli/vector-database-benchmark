from google.cloud import recommender_v1beta1

def sample_list_insight_types():
    if False:
        for i in range(10):
            print('nop')
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.ListInsightTypesRequest()
    page_result = client.list_insight_types(request=request)
    for response in page_result:
        print(response)