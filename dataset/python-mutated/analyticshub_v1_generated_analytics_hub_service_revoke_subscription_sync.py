from google.cloud import bigquery_analyticshub_v1

def sample_revoke_subscription():
    if False:
        return 10
    client = bigquery_analyticshub_v1.AnalyticsHubServiceClient()
    request = bigquery_analyticshub_v1.RevokeSubscriptionRequest(name='name_value')
    response = client.revoke_subscription(request=request)
    print(response)