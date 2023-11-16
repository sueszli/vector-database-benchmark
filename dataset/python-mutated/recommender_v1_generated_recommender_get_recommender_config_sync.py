from google.cloud import recommender_v1

def sample_get_recommender_config():
    if False:
        while True:
            i = 10
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.GetRecommenderConfigRequest(name='name_value')
    response = client.get_recommender_config(request=request)
    print(response)