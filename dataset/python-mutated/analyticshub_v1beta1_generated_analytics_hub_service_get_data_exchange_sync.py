from google.cloud import bigquery_data_exchange_v1beta1

def sample_get_data_exchange():
    if False:
        i = 10
        return i + 15
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.GetDataExchangeRequest(name='name_value')
    response = client.get_data_exchange(request=request)
    print(response)