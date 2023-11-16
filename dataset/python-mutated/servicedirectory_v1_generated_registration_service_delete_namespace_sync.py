from google.cloud import servicedirectory_v1

def sample_delete_namespace():
    if False:
        for i in range(10):
            print('nop')
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.DeleteNamespaceRequest(name='name_value')
    client.delete_namespace(request=request)