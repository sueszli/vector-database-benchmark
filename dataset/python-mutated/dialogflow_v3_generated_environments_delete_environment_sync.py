from google.cloud import dialogflowcx_v3

def sample_delete_environment():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.EnvironmentsClient()
    request = dialogflowcx_v3.DeleteEnvironmentRequest(name='name_value')
    client.delete_environment(request=request)