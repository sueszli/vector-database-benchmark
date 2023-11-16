from google.cloud import recommender_v1beta1

def sample_mark_recommendation_failed():
    if False:
        print('Hello World!')
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.MarkRecommendationFailedRequest(name='name_value', etag='etag_value')
    response = client.mark_recommendation_failed(request=request)
    print(response)