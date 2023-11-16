from google.cloud import recommender_v1beta1

def sample_list_recommenders():
    if False:
        return 10
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.ListRecommendersRequest()
    page_result = client.list_recommenders(request=request)
    for response in page_result:
        print(response)