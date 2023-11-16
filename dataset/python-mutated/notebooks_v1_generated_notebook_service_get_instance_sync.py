from google.cloud import notebooks_v1

def sample_get_instance():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)