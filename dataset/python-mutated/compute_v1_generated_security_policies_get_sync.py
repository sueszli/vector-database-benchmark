from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.GetSecurityPolicyRequest(project='project_value', security_policy='security_policy_value')
    response = client.get(request=request)
    print(response)