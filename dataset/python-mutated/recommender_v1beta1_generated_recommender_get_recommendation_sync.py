from google.cloud import recommender_v1beta1

def sample_get_recommendation():
    if False:
        while True:
            i = 10
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.GetRecommendationRequest(name='name_value')
    response = client.get_recommendation(request=request)
    print(response)