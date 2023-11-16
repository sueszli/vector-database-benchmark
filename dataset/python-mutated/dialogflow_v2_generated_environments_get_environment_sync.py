from google.cloud import dialogflow_v2

def sample_get_environment():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.EnvironmentsClient()
    request = dialogflow_v2.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)