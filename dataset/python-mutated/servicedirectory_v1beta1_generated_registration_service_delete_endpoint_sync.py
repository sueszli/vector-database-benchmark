from google.cloud import servicedirectory_v1beta1

def sample_delete_endpoint():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.DeleteEndpointRequest(name='name_value')
    client.delete_endpoint(request=request)