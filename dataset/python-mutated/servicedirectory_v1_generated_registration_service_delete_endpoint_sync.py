from google.cloud import servicedirectory_v1

def sample_delete_endpoint():
    if False:
        i = 10
        return i + 15
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.DeleteEndpointRequest(name='name_value')
    client.delete_endpoint(request=request)