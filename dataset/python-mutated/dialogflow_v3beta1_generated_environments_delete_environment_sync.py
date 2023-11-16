from google.cloud import dialogflowcx_v3beta1

def sample_delete_environment():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    request = dialogflowcx_v3beta1.DeleteEnvironmentRequest(name='name_value')
    client.delete_environment(request=request)