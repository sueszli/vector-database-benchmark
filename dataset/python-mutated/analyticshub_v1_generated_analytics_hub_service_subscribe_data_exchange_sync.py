from google.cloud import bigquery_analyticshub_v1

def sample_subscribe_data_exchange():
    if False:
        i = 10
        return i + 15
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.SubscribeDataExchangeRequest(name='name_value', destination='destination_value', subscription='subscription_value')
    operation = client.subscribe_data_exchange(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)