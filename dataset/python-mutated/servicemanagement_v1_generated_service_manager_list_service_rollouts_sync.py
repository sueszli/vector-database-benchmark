from google.cloud import servicemanagement_v1

def sample_list_service_rollouts():
    if False:
        while True:
            i = 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.ListServiceRolloutsRequest(service_name='service_name_value', filter='filter_value')
    page_result = client.list_service_rollouts(request=request)
    for response in page_result:
        print(response)