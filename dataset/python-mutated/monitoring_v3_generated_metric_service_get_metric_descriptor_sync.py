from google.cloud import monitoring_v3

def sample_get_metric_descriptor():
    if False:
        return 10
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.GetMetricDescriptorRequest(name='name_value')
    response = client.get_metric_descriptor(request=request)
    print(response)