from google.cloud import bigquery_analyticshub_v1

def sample_delete_listing():
    if False:
        print('Hello World!')
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.DeleteListingRequest(name='name_value')
    client.delete_listing(request=request)