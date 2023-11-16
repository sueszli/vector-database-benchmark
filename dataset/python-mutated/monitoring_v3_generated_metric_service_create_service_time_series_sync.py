from google.cloud import monitoring_v3

def sample_create_service_time_series():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.MetricServiceClient()
    request = monitoring_v3.CreateTimeSeriesRequest(name='name_value')
    client.create_service_time_series(request=request)