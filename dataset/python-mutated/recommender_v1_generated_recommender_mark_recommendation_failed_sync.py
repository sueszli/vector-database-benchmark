from google.cloud import recommender_v1

def sample_mark_recommendation_failed():
    if False:
        while True:
            i = 10
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.MarkRecommendationFailedRequest(name='name_value', etag='etag_value')
    response = client.mark_recommendation_failed(request=request)
    print(response)