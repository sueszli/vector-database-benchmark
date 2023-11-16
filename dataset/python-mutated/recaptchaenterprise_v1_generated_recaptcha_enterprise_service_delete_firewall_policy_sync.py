from google.cloud import recaptchaenterprise_v1

def sample_delete_firewall_policy():
    if False:
        for i in range(10):
            print('nop')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.DeleteFirewallPolicyRequest(name='name_value')
    client.delete_firewall_policy(request=request)