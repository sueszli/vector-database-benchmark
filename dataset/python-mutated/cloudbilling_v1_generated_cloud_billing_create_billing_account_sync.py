from google.cloud import billing_v1

def sample_create_billing_account():
    if False:
        i = 10
        return i + 15
    client = billing_v1.CloudBillingClient()
    request = billing_v1.CreateBillingAccountRequest()
    response = client.create_billing_account(request=request)
    print(response)