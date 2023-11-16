from google.cloud import recommendationengine_v1beta1

def sample_list_prediction_api_key_registrations():
    if False:
        return 10
    client = recommendationengine_v1beta1.PredictionApiKeyRegistryClient()
    request = recommendationengine_v1beta1.ListPredictionApiKeyRegistrationsRequest(parent='parent_value')
    page_result = client.list_prediction_api_key_registrations(request=request)
    for response in page_result:
        print(response)