from google.cloud import recaptchaenterprise_v1

def sample_list_firewall_policies():
    if False:
        print('Hello World!')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.ListFirewallPoliciesRequest(parent='parent_value')
    page_result = client.list_firewall_policies(request=request)
    for response in page_result:
        print(response)