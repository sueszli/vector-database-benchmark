from google.cloud import billing_v1

def sample_update_billing_account():
    if False:
        return 10
    client = billing_v1.CloudBillingClient()
    request = billing_v1.UpdateBillingAccountRequest(name='name_value')
    response = client.update_billing_account(request=request)
    print(response)