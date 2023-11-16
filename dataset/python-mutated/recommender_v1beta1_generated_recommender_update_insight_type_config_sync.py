from google.cloud import recommender_v1beta1

def sample_update_insight_type_config():
    if False:
        i = 10
        return i + 15
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.UpdateInsightTypeConfigRequest()
    response = client.update_insight_type_config(request=request)
    print(response)