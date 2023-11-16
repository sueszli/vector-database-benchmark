from google.cloud import notebooks_v1

def sample_get_runtime():
    if False:
        print('Hello World!')
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.GetRuntimeRequest(name='name_value')
    response = client.get_runtime(request=request)
    print(response)