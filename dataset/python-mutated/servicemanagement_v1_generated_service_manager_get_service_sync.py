from google.cloud import servicemanagement_v1

def sample_get_service():
    if False:
        return 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.GetServiceRequest(service_name='service_name_value')
    response = client.get_service(request=request)
    print(response)