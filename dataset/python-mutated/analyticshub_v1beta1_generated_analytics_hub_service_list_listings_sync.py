from google.cloud import bigquery_data_exchange_v1beta1

def sample_list_listings():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.ListListingsRequest(parent='parent_value')
    page_result = client.list_listings(request=request)
    for response in page_result:
        print(response)