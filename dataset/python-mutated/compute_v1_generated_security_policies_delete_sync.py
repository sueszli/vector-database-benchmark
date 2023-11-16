from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.DeleteSecurityPolicyRequest(project='project_value', security_policy='security_policy_value')
    response = client.delete(request=request)
    print(response)