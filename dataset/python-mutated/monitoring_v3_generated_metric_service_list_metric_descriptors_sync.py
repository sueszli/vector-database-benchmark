from google.cloud import monitoring_v3

def sample_list_metric_descriptors():
    if False:
        while True:
            i = 10
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.ListMetricDescriptorsRequest(name='name_value')
    page_result = client.list_metric_descriptors(request=request)
    for response in page_result:
        print(response)