from google.cloud import servicedirectory_v1beta1

def sample_create_namespace():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.CreateNamespaceRequest(parent='parent_value', namespace_id='namespace_id_value')
    response = client.create_namespace(request=request)
    print(response)