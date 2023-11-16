from google.cloud import recaptchaenterprise_v1

def sample_create_firewall_policy():
    if False:
        print('Hello World!')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.CreateFirewallPolicyRequest(parent='parent_value')
    response = client.create_firewall_policy(request=request)
    print(response)