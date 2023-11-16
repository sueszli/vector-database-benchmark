from google.cloud import dialogflow_v2beta1

def sample_delete_environment():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.EnvironmentsClient()
    request = dialogflow_v2beta1.DeleteEnvironmentRequest(name='name_value')
    client.delete_environment(request=request)