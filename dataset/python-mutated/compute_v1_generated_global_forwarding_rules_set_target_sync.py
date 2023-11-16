from google.cloud import compute_v1

def sample_set_target():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalForwardingRulesClient()
    request = compute_v1.SetTargetGlobalForwardingRuleRequest(forwarding_rule='forwarding_rule_value', project='project_value')
    response = client.set_target(request=request)
    print(response)