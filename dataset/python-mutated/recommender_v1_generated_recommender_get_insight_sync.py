from google.cloud import recommender_v1

def sample_get_insight():
    if False:
        return 10
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.GetInsightRequest(name='name_value')
    response = client.get_insight(request=request)
    print(response)