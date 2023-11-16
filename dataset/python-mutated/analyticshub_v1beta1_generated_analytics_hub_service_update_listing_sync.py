from google.cloud import bigquery_data_exchange_v1beta1

def sample_update_listing():
    if False:
        return 10
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    listing = bigquery_data_exchange_v1beta1.Listing()
    listing.display_name = 'display_name_value'
    request = bigquery_data_exchange_v1beta1.UpdateListingRequest(listing=listing)
    response = client.update_listing(request=request)
    print(response)