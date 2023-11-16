from google.cloud import servicedirectory_v1beta1

def sample_delete_namespace():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.DeleteNamespaceRequest(name='name_value')
    client.delete_namespace(request=request)