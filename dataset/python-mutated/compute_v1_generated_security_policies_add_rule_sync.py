from google.cloud import compute_v1

def sample_add_rule():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.AddRuleSecurityPolicyRequest(project='project_value', security_policy='security_policy_value')
    response = client.add_rule(request=request)
    print(response)