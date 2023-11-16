from google.cloud import servicedirectory_v1

def sample_get_service():
    if False:
        return 10
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)