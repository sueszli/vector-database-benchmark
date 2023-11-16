from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalForwardingRulesClient()
    request = compute_v1.PatchGlobalForwardingRuleRequest(forwarding_rule='forwarding_rule_value', project='project_value')
    response = client.patch(request=request)
    print(response)