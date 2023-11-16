from google.cloud import filestore_v1

def sample_get_instance():
    if False:
        return 10
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)