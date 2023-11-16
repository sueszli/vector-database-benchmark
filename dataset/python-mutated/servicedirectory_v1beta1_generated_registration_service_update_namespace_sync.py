from google.cloud import servicedirectory_v1beta1

def sample_update_namespace():
    if False:
        for i in range(10):
            print('nop')
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.UpdateNamespaceRequest()
    response = client.update_namespace(request=request)
    print(response)