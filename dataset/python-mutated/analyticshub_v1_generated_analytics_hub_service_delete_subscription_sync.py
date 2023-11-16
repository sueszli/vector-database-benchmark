from google.cloud import bigquery_analyticshub_v1

def sample_delete_subscription():
    if False:
        print('Hello World!')
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.DeleteSubscriptionRequest(name='name_value')
    operation = client.delete_subscription(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)