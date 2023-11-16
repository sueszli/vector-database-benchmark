from google.cloud import notebooks_v1beta1

def sample_get_environment():
    if False:
        return 10
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)