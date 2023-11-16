from google.cloud import bigquery_data_exchange_v1beta1

def sample_create_data_exchange():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    data_exchange = bigquery_data_exchange_v1beta1.DataExchange()
    data_exchange.display_name = 'display_name_value'
    request = bigquery_data_exchange_v1beta1.CreateDataExchangeRequest(parent='parent_value', data_exchange_id='data_exchange_id_value', data_exchange=data_exchange)
    response = client.create_data_exchange(request=request)
    print(response)