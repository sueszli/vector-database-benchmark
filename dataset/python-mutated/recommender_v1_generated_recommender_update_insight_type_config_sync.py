from google.cloud import recommender_v1

def sample_update_insight_type_config():
    if False:
        for i in range(10):
            print('nop')
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.UpdateInsightTypeConfigRequest()
    response = client.update_insight_type_config(request=request)
    print(response)