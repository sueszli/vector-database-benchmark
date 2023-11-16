from google.cloud import bigquery_analyticshub_v1

def sample_update_data_exchange():
    if False:
        i = 10
        return i + 15
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    data_exchange = bigquery_analyticshub_v1.DataExchange()
    data_exchange.display_name = 'display_name_value'
    request = bigquery_analyticshub_v1.UpdateDataExchangeRequest(data_exchange=data_exchange)
    response = client.update_data_exchange(request=request)
    print(response)