from google.cloud import bigquery_data_exchange_v1beta1

def sample_list_data_exchanges():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.ListDataExchangesRequest(parent='parent_value')
    page_result = client.list_data_exchanges(request=request)
    for response in page_result:
        print(response)