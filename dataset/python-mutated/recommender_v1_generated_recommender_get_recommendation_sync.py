from google.cloud import recommender_v1

def sample_get_recommendation():
    if False:
        i = 10
        return i + 15
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.GetRecommendationRequest(name='name_value')
    response = client.get_recommendation(request=request)
    print(response)