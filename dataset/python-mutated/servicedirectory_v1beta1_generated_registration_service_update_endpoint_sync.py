from google.cloud import servicedirectory_v1beta1

def sample_update_endpoint():
    if False:
        return 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.UpdateEndpointRequest()
    response = client.update_endpoint(request=request)
    print(response)