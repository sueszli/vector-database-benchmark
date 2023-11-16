from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalForwardingRulesClient()
    request = compute_v1.GetGlobalForwardingRuleRequest(forwarding_rule='forwarding_rule_value', project='project_value')
    response = client.get(request=request)
    print(response)