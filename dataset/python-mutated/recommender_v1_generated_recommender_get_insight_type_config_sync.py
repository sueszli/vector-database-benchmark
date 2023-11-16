from google.cloud import recommender_v1

def sample_get_insight_type_config():
    if False:
        while True:
            i = 10
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.GetInsightTypeConfigRequest(name='name_value')
    response = client.get_insight_type_config(request=request)
    print(response)