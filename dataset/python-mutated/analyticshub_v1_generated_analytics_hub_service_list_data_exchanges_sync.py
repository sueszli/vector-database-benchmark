from google.cloud import bigquery_analyticshub_v1

def sample_list_data_exchanges():
    if False:
        return 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.ListDataExchangesRequest(parent='parent_value')
    page_result = client.list_data_exchanges(request=request)
    for response in page_result:
        print(response)