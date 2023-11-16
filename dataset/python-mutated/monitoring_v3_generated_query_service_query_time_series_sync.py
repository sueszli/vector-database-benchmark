from google.cloud import monitoring_v3

def sample_query_time_series():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.QueryServiceClient()
    request = monitoring_v3.QueryTimeSeriesRequest(name='name_value', query='query_value')
    page_result = client.query_time_series(request=request)
    for response in page_result:
        print(response)