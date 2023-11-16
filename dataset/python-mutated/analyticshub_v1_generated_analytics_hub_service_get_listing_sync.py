from google.cloud import bigquery_analyticshub_v1

def sample_get_listing():
    if False:
        while True:
            i = 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.GetListingRequest(name='name_value')
    response = client.get_listing(request=request)
    print(response)