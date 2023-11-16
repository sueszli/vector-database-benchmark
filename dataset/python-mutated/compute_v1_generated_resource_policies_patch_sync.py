from google.cloud import compute_v1

def sample_patch():
    if False:
        while True:
            i = 10
    client = compute_v1.ResourcePoliciesClient()
    request = compute_v1.PatchResourcePolicyRequest(project='project_value', region='region_value', resource_policy='resource_policy_value')
    response = client.patch(request=request)
    print(response)