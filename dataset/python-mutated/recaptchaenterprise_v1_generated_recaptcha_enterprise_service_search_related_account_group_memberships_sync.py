from google.cloud import recaptchaenterprise_v1

def sample_search_related_account_group_memberships():
    if False:
        while True:
            i = 10
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.SearchRelatedAccountGroupMembershipsRequest(project='project_value')
    page_result = client.search_related_account_group_memberships(request=request)
    for response in page_result:
        print(response)