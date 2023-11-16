from google.cloud import notebooks_v2

def sample_get_instance():
    if False:
        return 10
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)