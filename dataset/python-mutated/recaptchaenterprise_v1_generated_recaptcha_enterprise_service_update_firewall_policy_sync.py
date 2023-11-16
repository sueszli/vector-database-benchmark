from google.cloud import recaptchaenterprise_v1

def sample_update_firewall_policy():
    if False:
        return 10
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.UpdateFirewallPolicyRequest()
    response = client.update_firewall_policy(request=request)
    print(response)