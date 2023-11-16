from google.cloud import servicedirectory_v1

def sample_get_namespace():
    if False:
        print('Hello World!')
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.GetNamespaceRequest(name='name_value')
    response = client.get_namespace(request=request)
    print(response)