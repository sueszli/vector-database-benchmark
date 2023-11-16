from google.cloud import servicemanagement_v1

def sample_list_service_configs():
    if False:
        print('Hello World!')
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.ListServiceConfigsRequest(service_name='service_name_value')
    page_result = client.list_service_configs(request=request)
    for response in page_result:
        print(response)