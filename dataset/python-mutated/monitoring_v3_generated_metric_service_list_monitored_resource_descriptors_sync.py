from google.cloud import monitoring_v3

def sample_list_monitored_resource_descriptors():
    if False:
        print('Hello World!')
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.ListMonitoredResourceDescriptorsRequest(name='name_value')
    page_result = client.list_monitored_resource_descriptors(request=request)
    for response in page_result:
        print(response)