from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.ResourcePoliciesClient()
    request = compute_v1.DeleteResourcePolicyRequest(project='project_value', region='region_value', resource_policy='resource_policy_value')
    response = client.delete(request=request)
    print(response)