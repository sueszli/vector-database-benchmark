from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SslPoliciesClient()
    request = compute_v1.PatchSslPolicyRequest(project='project_value', ssl_policy='ssl_policy_value')
    response = client.patch(request=request)
    print(response)