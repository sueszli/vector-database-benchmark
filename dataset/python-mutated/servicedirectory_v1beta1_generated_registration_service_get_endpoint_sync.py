from google.cloud import servicedirectory_v1beta1

def sample_get_endpoint():
    if False:
        print('Hello World!')
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.GetEndpointRequest(name='name_value')
    response = client.get_endpoint(request=request)
    print(response)