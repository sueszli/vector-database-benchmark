from google.cloud import servicedirectory_v1

def sample_delete_service():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.DeleteServiceRequest(name='name_value')
    client.delete_service(request=request)