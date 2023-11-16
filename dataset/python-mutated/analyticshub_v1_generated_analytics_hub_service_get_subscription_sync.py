from google.cloud import bigquery_analyticshub_v1

def sample_get_subscription():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.GetSubscriptionRequest(name='name_value')
    response = client.get_subscription(request=request)
    print(response)