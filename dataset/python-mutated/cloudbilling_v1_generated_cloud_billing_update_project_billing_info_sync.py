from google.cloud import billing_v1

def sample_update_project_billing_info():
    if False:
        i = 10
        return i + 15
    client = billing_v1.CloudBillingClient()
    request = billing_v1.UpdateProjectBillingInfoRequest(name='name_value')
    response = client.update_project_billing_info(request=request)
    print(response)