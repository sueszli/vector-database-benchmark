from google.cloud import monitoring_v3

def sample_create_metric_descriptor():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.CreateMetricDescriptorRequest(name='name_value')
    response = client.create_metric_descriptor(request=request)
    print(response)