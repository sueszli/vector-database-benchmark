from google.cloud import recommender_v1beta1

def sample_mark_recommendation_succeeded():
    if False:
        i = 10
        return i + 15
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.MarkRecommendationSucceededRequest(name='name_value', etag='etag_value')
    response = client.mark_recommendation_succeeded(request=request)
    print(response)