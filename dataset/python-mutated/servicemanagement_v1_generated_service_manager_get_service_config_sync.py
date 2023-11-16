from google.cloud import servicemanagement_v1

def sample_get_service_config():
    if False:
        while True:
            i = 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.GetServiceConfigRequest(service_name='service_name_value', config_id='config_id_value')
    response = client.get_service_config(request=request)
    print(response)