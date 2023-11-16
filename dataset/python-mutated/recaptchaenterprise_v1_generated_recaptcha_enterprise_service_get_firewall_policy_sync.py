from google.cloud import recaptchaenterprise_v1

def sample_get_firewall_policy():
    if False:
        for i in range(10):
            print('nop')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.GetFirewallPolicyRequest(name='name_value')
    response = client.get_firewall_policy(request=request)
    print(response)