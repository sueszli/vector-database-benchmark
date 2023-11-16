from google.cloud import servicedirectory_v1beta1

def sample_delete_service():
    if False:
        for i in range(10):
            print('nop')
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.DeleteServiceRequest(name='name_value')
    client.delete_service(request=request)