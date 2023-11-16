from google.cloud import bigquery_data_exchange_v1beta1

def sample_list_org_data_exchanges():
    if False:
        while True:
            i = 10
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = bigquery_data_exchange_v1beta1.ListOrgDataExchangesRequest(organization='organization_value')
    page_result = client.list_org_data_exchanges(request=request)
    for response in page_result:
        print(response)