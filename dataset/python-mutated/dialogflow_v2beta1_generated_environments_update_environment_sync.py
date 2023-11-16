from google.cloud import dialogflow_v2beta1

def sample_update_environment():
    if False:
        return 10
    client = dialogflow_v2beta1.EnvironmentsClient()
    request = dialogflow_v2beta1.UpdateEnvironmentRequest()
    response = client.update_environment(request=request)
    print(response)