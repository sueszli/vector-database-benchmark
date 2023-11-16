from google.cloud import bigquery_data_exchange_v1beta1

def sample_delete_data_exchange():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.DeleteDataExchangeRequest(name='name_value')
    client.delete_data_exchange(request=request)