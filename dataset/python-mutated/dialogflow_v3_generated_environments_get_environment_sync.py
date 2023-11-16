from google.cloud import dialogflowcx_v3

def sample_get_environment():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.EnvironmentsClient()
    request = dialogflowcx_v3.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)