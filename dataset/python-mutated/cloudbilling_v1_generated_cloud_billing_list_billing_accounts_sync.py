from google.cloud import billing_v1

def sample_list_billing_accounts():
    if False:
        for i in range(10):
            print('nop')
    client = billing_v1.CloudBillingClient()
    request = billing_v1.ListBillingAccountsRequest()
    page_result = client.list_billing_accounts(request=request)
    for response in page_result:
        print(response)