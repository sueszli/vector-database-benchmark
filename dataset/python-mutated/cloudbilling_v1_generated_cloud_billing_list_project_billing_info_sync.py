from google.cloud import billing_v1

def sample_list_project_billing_info():
    if False:
        for i in range(10):
            print('nop')
    client = billing_v1.CloudBillingClient()
    request = billing_v1.ListProjectBillingInfoRequest(name='name_value')
    page_result = client.list_project_billing_info(request=request)
    for response in page_result:
        print(response)