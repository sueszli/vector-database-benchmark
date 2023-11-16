from google.cloud import billing_v1

def sample_get_project_billing_info():
    if False:
        print('Hello World!')
    client = billing_v1.CloudBillingClient()
    request = billing_v1.GetProjectBillingInfoRequest(name='name_value')
    response = client.get_project_billing_info(request=request)
    print(response)