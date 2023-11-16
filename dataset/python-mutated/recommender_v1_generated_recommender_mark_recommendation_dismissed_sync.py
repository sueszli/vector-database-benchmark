from google.cloud import recommender_v1

def sample_mark_recommendation_dismissed():
    if False:
        return 10
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.MarkRecommendationDismissedRequest(name='name_value')
    response = client.mark_recommendation_dismissed(request=request)
    print(response)