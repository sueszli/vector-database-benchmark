from google.cloud import notebooks_v1

def sample_get_environment():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)