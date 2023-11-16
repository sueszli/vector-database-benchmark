from google.cloud import monitoring_v3

def sample_list_service_level_objectives():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.ListServiceLevelObjectivesRequest(parent='parent_value')
    page_result = client.list_service_level_objectives(request=request)
    for response in page_result:
        print(response)