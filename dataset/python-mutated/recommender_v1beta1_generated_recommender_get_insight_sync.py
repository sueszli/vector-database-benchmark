from google.cloud import recommender_v1beta1

def sample_get_insight():
    if False:
        i = 10
        return i + 15
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.GetInsightRequest(name='name_value')
    response = client.get_insight(request=request)
    print(response)