from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.ResourcePoliciesClient()
    request = compute_v1.GetResourcePolicyRequest(project='project_value', region='region_value', resource_policy='resource_policy_value')
    response = client.get(request=request)
    print(response)