from google.cloud import dialogflow_v2

def sample_get_validation_result():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.GetValidationResultRequest(parent='parent_value')
    response = client.get_validation_result(request=request)
    print(response)