from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.GlobalForwardingRulesClient()
    request = compute_v1.DeleteGlobalForwardingRuleRequest(forwarding_rule='forwarding_rule_value', project='project_value')
    response = client.delete(request=request)
    print(response)