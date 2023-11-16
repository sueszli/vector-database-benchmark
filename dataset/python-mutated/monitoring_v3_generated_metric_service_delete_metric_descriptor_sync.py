from google.cloud import monitoring_v3

def sample_delete_metric_descriptor():
    if False:
        print('Hello World!')
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.DeleteMetricDescriptorRequest(name='name_value')
    client.delete_metric_descriptor(request=request)