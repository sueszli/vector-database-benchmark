from google.cloud import servicedirectory_v1

def sample_update_endpoint():
    if False:
        i = 10
        return i + 15
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.UpdateEndpointRequest()
    response = client.update_endpoint(request=request)
    print(response)