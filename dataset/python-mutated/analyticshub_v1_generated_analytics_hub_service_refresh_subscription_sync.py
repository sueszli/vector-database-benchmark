from google.cloud import bigquery_analyticshub_v1

def sample_refresh_subscription():
    if False:
        while True:
            i = 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.RefreshSubscriptionRequest(name='name_value')
    operation = client.refresh_subscription(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)