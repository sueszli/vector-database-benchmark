from google.cloud import dialogflow_v2beta1

def sample_get_validation_result():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.AgentsClient()
    request = dialogflow_v2beta1.GetValidationResultRequest(parent='parent_value')
    response = client.get_validation_result(request=request)
    print(response)