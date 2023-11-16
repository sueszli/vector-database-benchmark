from google.cloud import bigquery_analyticshub_v1

def sample_update_listing():
    if False:
        return 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    listing = bigquery_analyticshub_v1.Listing()
    listing.display_name = 'display_name_value'
    request = bigquery_analyticshub_v1.UpdateListingRequest(listing=listing)
    response = client.update_listing(request=request)
    print(response)