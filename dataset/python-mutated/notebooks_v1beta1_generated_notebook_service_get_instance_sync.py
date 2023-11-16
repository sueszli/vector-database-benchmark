from google.cloud import notebooks_v1beta1

def sample_get_instance():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)