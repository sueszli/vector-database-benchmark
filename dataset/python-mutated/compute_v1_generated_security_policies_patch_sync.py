from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.PatchSecurityPolicyRequest(project='project_value', security_policy='security_policy_value')
    response = client.patch(request=request)
    print(response)