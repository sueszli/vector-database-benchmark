from google.cloud import monitoring_v3

def sample_get_monitored_resource_descriptor():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.GetMonitoredResourceDescriptorRequest(name='name_value')
    response = client.get_monitored_resource_descriptor(request=request)
    print(response)