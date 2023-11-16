from google.cloud import recommendationengine_v1beta1

def sample_delete_prediction_api_key_registration():
    if False:
        print('Hello World!')
    client = recommendationengine_v1beta1.PredictionApiKeyRegistryClient()
    request = recommendationengine_v1beta1.DeletePredictionApiKeyRegistrationRequest(name='name_value')
    client.delete_prediction_api_key_registration(request=request)