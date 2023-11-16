from google.cloud import compute_v1

def sample_patch_rule():
    if False:
        print('Hello World!')
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.PatchRuleSecurityPolicyRequest(project='project_value', security_policy='security_policy_value')
    response = client.patch_rule(request=request)
    print(response)