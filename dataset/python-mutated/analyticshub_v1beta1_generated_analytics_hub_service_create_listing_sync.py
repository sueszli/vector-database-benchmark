from google.cloud import bigquery_data_exchange_v1beta1

def sample_create_listing():
    if False:
        i = 10
        return i + 15
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    listing = bigquery_data_exchange_v1beta1.Listing()
    listing.display_name = 'display_name_value'
    request = bigquery_data_exchange_v1beta1.CreateListingRequest(parent='parent_value', listing_id='listing_id_value', listing=listing)
    response = client.create_listing(request=request)
    print(response)