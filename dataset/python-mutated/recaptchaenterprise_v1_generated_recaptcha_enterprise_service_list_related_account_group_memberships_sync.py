from google.cloud import recaptchaenterprise_v1

def sample_list_related_account_group_memberships():
    if False:
        return 10
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.ListRelatedAccountGroupMembershipsRequest(parent='parent_value')
    page_result = client.list_related_account_group_memberships(request=request)
    for response in page_result:
        print(response)