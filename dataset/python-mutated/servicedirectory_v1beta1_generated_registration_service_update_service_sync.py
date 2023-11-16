from google.cloud import servicedirectory_v1beta1

def sample_update_service():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.UpdateServiceRequest()
    response = client.update_service(request=request)
    print(response)