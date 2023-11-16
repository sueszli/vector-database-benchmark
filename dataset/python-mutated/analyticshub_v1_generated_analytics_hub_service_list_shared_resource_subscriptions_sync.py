from google.cloud import bigquery_analyticshub_v1

def sample_list_shared_resource_subscriptions():
    if False:
        while True:
            i = 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.ListSharedResourceSubscriptionsRequest(resource='resource_value')
    page_result = client.list_shared_resource_subscriptions(request=request)
    for response in page_result:
        print(response)