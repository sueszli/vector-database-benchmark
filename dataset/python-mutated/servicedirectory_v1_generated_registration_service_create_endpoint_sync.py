from google.cloud import servicedirectory_v1

def sample_create_endpoint():
    if False:
        i = 10
        return i + 15
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.CreateEndpointRequest(parent='parent_value', endpoint_id='endpoint_id_value')
    response = client.create_endpoint(request=request)
    print(response)