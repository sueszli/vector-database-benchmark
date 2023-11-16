from google.cloud import compute_v1

def sample_insert():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.InsertSecurityPolicyRequest(project='project_value')
    response = client.insert(request=request)
    print(response)