from google.cloud import bigquery_data_exchange_v1beta1

def sample_get_listing():
    if False:
        return 10
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.GetListingRequest(name='name_value')
    response = client.get_listing(request=request)
    print(response)