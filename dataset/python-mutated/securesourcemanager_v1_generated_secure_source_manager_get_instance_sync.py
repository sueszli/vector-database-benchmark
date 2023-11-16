from google.cloud import securesourcemanager_v1

def sample_get_instance():
    if False:
        for i in range(10):
            print('nop')
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)