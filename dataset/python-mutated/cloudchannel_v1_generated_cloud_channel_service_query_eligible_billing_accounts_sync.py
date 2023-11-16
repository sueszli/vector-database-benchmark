from google.cloud import channel_v1

def sample_query_eligible_billing_accounts():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.QueryEligibleBillingAccountsRequest(customer='customer_value', skus=['skus_value1', 'skus_value2'])
    response = client.query_eligible_billing_accounts(request=request)
    print(response)