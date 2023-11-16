from google.cloud import servicedirectory_v1beta1

def sample_create_endpoint():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.CreateEndpointRequest(parent='parent_value', endpoint_id='endpoint_id_value')
    response = client.create_endpoint(request=request)
    print(response)