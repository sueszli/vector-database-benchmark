from google.cloud import dialogflow_v2beta1

def sample_get_environment():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.EnvironmentsClient()
    request = dialogflow_v2beta1.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)