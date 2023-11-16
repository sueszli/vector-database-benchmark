from google.cloud import monitoring_v3

def sample_create_time_series():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.CreateTimeSeriesRequest(name='name_value')
    client.create_time_series(request=request)