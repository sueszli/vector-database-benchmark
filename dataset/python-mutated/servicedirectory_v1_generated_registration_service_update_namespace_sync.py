from google.cloud import servicedirectory_v1

def sample_update_namespace():
    if False:
        print('Hello World!')
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.UpdateNamespaceRequest()
    response = client.update_namespace(request=request)
    print(response)