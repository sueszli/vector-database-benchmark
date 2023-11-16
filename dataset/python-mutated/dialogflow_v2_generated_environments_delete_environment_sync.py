from google.cloud import dialogflow_v2

def sample_delete_environment():
    if False:
        return 10
    client = dialogflow_v2.EnvironmentsClient()
    request = dialogflow_v2.DeleteEnvironmentRequest(name='name_value')
    client.delete_environment(request=request)