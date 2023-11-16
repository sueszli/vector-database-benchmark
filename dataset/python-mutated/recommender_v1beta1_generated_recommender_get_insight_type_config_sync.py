from google.cloud import recommender_v1beta1

def sample_get_insight_type_config():
    if False:
        i = 10
        return i + 15
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.GetInsightTypeConfigRequest(name='name_value')
    response = client.get_insight_type_config(request=request)
    print(response)