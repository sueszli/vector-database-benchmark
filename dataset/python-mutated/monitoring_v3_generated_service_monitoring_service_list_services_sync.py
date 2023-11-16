from google.cloud import monitoring_v3

def sample_list_services():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.ListServicesRequest(parent='parent_value')
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)