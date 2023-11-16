from google.cloud import bigquery_analyticshub_v1

def sample_list_listings():
    if False:
        return 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.ListListingsRequest(parent='parent_value')
    page_result = client.list_listings(request=request)
    for response in page_result:
        print(response)