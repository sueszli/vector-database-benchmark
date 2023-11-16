from google.cloud import servicedirectory_v1beta1

def sample_get_service():
    if False:
        i = 10
        return i + 15
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)