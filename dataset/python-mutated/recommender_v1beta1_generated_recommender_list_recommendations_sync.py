from google.cloud import recommender_v1beta1

def sample_list_recommendations():
    if False:
        return 10
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.ListRecommendationsRequest(parent='parent_value')
    page_result = client.list_recommendations(request=request)
    for response in page_result:
        print(response)