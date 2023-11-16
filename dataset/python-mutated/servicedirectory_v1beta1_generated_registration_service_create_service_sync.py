from google.cloud import servicedirectory_v1beta1

def sample_create_service():
    if False:
        while True:
            i = 10
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.CreateServiceRequest(parent='parent_value', service_id='service_id_value')
    response = client.create_service(request=request)
    print(response)