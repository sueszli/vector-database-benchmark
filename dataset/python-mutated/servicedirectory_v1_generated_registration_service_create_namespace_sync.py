from google.cloud import servicedirectory_v1

def sample_create_namespace():
    if False:
        for i in range(10):
            print('nop')
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.CreateNamespaceRequest(parent='parent_value', namespace_id='namespace_id_value')
    response = client.create_namespace(request=request)
    print(response)