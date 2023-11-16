from google.cloud import servicedirectory_v1beta1

def sample_get_namespace():
    if False:
        return 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.GetNamespaceRequest(name='name_value')
    response = client.get_namespace(request=request)
    print(response)