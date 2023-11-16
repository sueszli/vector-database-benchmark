from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionSslPoliciesClient()
    request = compute_v1.PatchRegionSslPolicyRequest(project='project_value', region='region_value', ssl_policy='ssl_policy_value')
    response = client.patch(request=request)
    print(response)