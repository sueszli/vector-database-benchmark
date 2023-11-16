from google.cloud import servicemanagement_v1

def sample_create_service_config():
    if False:
        return 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.CreateServiceConfigRequest(service_name='service_name_value')
    response = client.create_service_config(request=request)
    print(response)