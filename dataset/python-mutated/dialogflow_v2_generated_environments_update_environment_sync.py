from google.cloud import dialogflow_v2

def sample_update_environment():
    if False:
        return 10
    client = dialogflow_v2.EnvironmentsClient()
    request = dialogflow_v2.UpdateEnvironmentRequest()
    response = client.update_environment(request=request)
    print(response)