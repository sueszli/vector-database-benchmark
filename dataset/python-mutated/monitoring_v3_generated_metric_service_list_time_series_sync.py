from google.cloud import monitoring_v3

def sample_list_time_series():
    if False:
        print('Hello World!')
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.ListTimeSeriesRequest(name='name_value', filter='filter_value', view='HEADERS')
    page_result = client.list_time_series(request=request)
    for response in page_result:
        print(response)