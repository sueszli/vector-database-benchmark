from google.cloud import bigquery_analyticshub_v1

def sample_delete_data_exchange():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.DeleteDataExchangeRequest(name='name_value')
    client.delete_data_exchange(request=request)