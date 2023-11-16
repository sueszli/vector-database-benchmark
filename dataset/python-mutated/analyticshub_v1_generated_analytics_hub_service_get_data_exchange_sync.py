from google.cloud import bigquery_analyticshub_v1

def sample_get_data_exchange():
    if False:
        while True:
            i = 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.GetDataExchangeRequest(name='name_value')
    response = client.get_data_exchange(request=request)
    print(response)