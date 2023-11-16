from google.cloud import recaptchaenterprise_v1

def sample_list_keys():
    if False:
        i = 10
        return i + 15
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.ListKeysRequest(parent='parent_value')
    page_result = client.list_keys(request=request)
    for response in page_result:
        print(response)