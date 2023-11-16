from google.cloud import securesourcemanager_v1

def sample_get_repository():
    if False:
        print('Hello World!')
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.GetRepositoryRequest(name='name_value')
    response = client.get_repository(request=request)
    print(response)