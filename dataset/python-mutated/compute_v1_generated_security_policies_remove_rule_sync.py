from google.cloud import compute_v1

def sample_remove_rule():
    if False:
        return 10
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.RemoveRuleSecurityPolicyRequest(project='project_value', security_policy='security_policy_value')
    response = client.remove_rule(request=request)
    print(response)