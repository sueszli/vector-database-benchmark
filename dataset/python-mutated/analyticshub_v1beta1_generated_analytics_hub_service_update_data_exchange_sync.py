from google.cloud import bigquery_data_exchange_v1beta1

def sample_update_data_exchange():
    if False:
        return 10
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    data_exchange = bigquery_data_exchange_v1beta1.DataExchange()
    data_exchange.display_name = 'display_name_value'
    request = bigquery_data_exchange_v1beta1.UpdateDataExchangeRequest(data_exchange=data_exchange)
    response = client.update_data_exchange(request=request)
    print(response)