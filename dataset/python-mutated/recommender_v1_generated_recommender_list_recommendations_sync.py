from google.cloud import recommender_v1

def sample_list_recommendations():
    if False:
        for i in range(10):
            print('nop')
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.ListRecommendationsRequest(parent='parent_value')
    page_result = client.list_recommendations(request=request)
    for response in page_result:
        print(response)