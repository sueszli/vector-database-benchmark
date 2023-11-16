from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.ForwardingRulesClient()
    request = compute_v1.PatchForwardingRuleRequest(forwarding_rule='forwarding_rule_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)