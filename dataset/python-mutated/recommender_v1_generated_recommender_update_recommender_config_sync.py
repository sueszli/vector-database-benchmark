from google.cloud import recommender_v1

def sample_update_recommender_config():
    if False:
        while True:
            i = 10
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.UpdateRecommenderConfigRequest()
    response = client.update_recommender_config(request=request)
    print(response)