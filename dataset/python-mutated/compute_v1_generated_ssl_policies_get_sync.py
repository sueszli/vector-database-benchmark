from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.SslPoliciesClient()
    request = compute_v1.GetSslPolicyRequest(project='project_value', ssl_policy='ssl_policy_value')
    response = client.get(request=request)
    print(response)