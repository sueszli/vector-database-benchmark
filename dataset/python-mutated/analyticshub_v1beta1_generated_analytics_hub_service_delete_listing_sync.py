from google.cloud import bigquery_data_exchange_v1beta1

def sample_delete_listing():
    if False:
        print('Hello World!')
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.DeleteListingRequest(name='name_value')
    client.delete_listing(request=request)