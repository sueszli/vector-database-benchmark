from google.cloud import recaptchaenterprise_v1

def sample_list_related_account_groups():
    if False:
        print('Hello World!')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.ListRelatedAccountGroupsRequest(parent='parent_value')
    page_result = client.list_related_account_groups(request=request)
    for response in page_result:
        print(response)