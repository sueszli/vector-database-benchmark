from google.cloud import notebooks_v1

def sample_get_instance_health():
    if False:
        return 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.GetInstanceHealthRequest(name='name_value')
    response = client.get_instance_health(request=request)
    print(response)