from google.cloud import recommender_v1beta1

def sample_update_recommender_config():
    if False:
        for i in range(10):
            print('nop')
    client = recommender_v1beta1.RecommenderClient()
    request = recommender_v1beta1.UpdateRecommenderConfigRequest()
    response = client.update_recommender_config(request=request)
    print(response)