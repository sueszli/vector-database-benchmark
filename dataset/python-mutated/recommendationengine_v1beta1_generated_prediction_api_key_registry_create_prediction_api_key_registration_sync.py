from google.cloud import recommendationengine_v1beta1

def sample_create_prediction_api_key_registration():
    if False:
        for i in range(10):
            print('nop')
    client = recommendationengine_v1beta1.PredictionApiKeyRegistryClient()
    request = recommendationengine_v1beta1.CreatePredictionApiKeyRegistrationRequest(parent='parent_value')
    response = client.create_prediction_api_key_registration(request=request)
    print(response)