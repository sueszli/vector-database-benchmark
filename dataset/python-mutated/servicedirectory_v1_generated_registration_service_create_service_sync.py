from google.cloud import servicedirectory_v1

def sample_create_service():
    if False:
        print('Hello World!')
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.CreateServiceRequest(parent='parent_value', service_id='service_id_value')
    response = client.create_service(request=request)
    print(response)